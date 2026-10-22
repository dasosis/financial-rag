# LLM gateway

An OpenAI-compatible API between the app's Python API and the model providers.
The app only changes its base URL (`LLM_GATEWAY_URL`, see `financial_rag/config.py`);
with it unset, the app calls OpenAI directly as before. The main README's
"LLM gateway" section covers running it with Docker.

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | plain and streamed (SSE), passed through to the provider |
| `POST /v1/embeddings` | passed through |
| `GET /healthz` | the process is up |
| `GET /readyz` | it takes requests (503 while shutting down, so the load balancer moves on) |
| `GET /metrics` | Prometheus, on a separate internal port (`GATEWAY_METRICS_ADDR`, `:9090`) |

Every `/v1` call needs `Authorization: Bearer $LLM_GATEWAY_KEY` and an
`X-Consumer-Id` (`user:<id>` or `system:<name>`), or it gets 401. Errors are
OpenAI-shaped (`{"error": {"message", "type", "code"}}`).

Streams are relayed event by event and flushed at once. The gateway always asks
the provider for the usage chunk (`stream_options.include_usage`), counts its
tokens, and drops it again unless the caller asked for it.

## Limits and budgets

Before a call goes to a provider, one Lua script on the shared Redis
(`GATEWAY_LIMITS_REDIS_URL`) checks, atomically, and takes from:
- the consumer's **token bucket** (`user:` consumers only; ingest and the
  benchmarks are jobs, paced by the global bucket alone);
- the **global bucket**, all consumers together;
- the consumer's **daily budget** (UTC days), which the call reserves an estimate
  from: its prompt at about four characters a token, plus `max_tokens` or
  `GATEWAY_COMPLETION_ESTIMATE`. Once the provider reports usage, a second script
  corrects the reservation to the actual cost; a failed call is refunded, and a
  stream cut off before its usage chunk keeps the estimate.

Every script reads the time from Redis (`TIME`), never a replica's clock. A
refusal is `429` with `Retry-After` and `x-should-retry: false`, `error.code`
`rate_limited` or `budget_exhausted`; the app answers its user with its own usual
429 and refunds the question.

Budgets count **weighted tokens**: one unit is the price of a gpt-4o-mini input
token, so its output tokens count 4, cached input 0.5, an embedding token 0.13
(`internal/pricing`; `GATEWAY_PRICES` adds or overrides models, and an unknown
model is charged as gpt-4o). The default, 320,000 a day, is about 20 questions
at the ~16,000 input tokens a question was measured to use.

The limits' Redis runs with `noeviction`. If it can't be reached, each replica
falls back to limits of its own at a third of the configured ones
(`GATEWAY_REPLICAS`), sets `gateway_limiter_degraded` to 1, logs it once, and
goes back to Redis as soon as it answers a ping.

## Response cache

Repeats of a temperature-0 chat call are answered from a Redis of their own
(`GATEWAY_CACHE_REDIS_URL`; `allkeys-lru`, 64 MB, entries expire after 7 days)
instead of the provider, at no cost to the consumer's budget or rate.

- **Exact matches only.** The key is the SHA-256 of the canonical JSON (sorted
  keys, one spelling per number) of every field that can change the answer:
  model, messages, temperature, tools, response_format, max_tokens, seed, top_p
  and the rest. Only delivery fields (`stream`, `stream_options`, `user`,
  `metadata`, `store`, …) are left out. Never "similar" questions: "revenue in
  2023" and "revenue in 2024" embed almost identically, and a semantic cache would
  answer one with the other's figures.
- **Only temperature 0 and one answer** (`n` absent or 1); anything else is meant to vary.
- **Only answers that finished normally** (`stop`, `tool_calls`): never an error,
  a stream that broke off, or an answer cut at `max_tokens`. A streamed answer is
  assembled into the plain JSON form before it is stored, and a streamed hit is
  replayed as SSE from it (with the usage chunk if the caller asked for one), so
  either kind of call can answer the other.
- **`Cache-Control: no-cache`** skips the lookup (the benchmarks send it: a cached
  answer would score the cache); `no-store` also keeps the answer out.
- Only the response body is stored, under a hash: never the prompt. Prompts are
  never logged either.
- A cache that is down or slow is a miss; the call goes to the provider.

`X-Gateway-Cache` says `hit`, `miss` or `bypass`.
`gateway_cache_requests_total{result}` gives the hit ratio, and
`gateway_cache_saved_usd_total{model}` what the hits would have cost.

## Providers, circuit breakers and fallback

Calls go to OpenAI first and, if `OPENROUTER_API_KEY` is set, to OpenRouter when
OpenAI fails, under the backup's name for the model (`GATEWAY_BACKUP_MODELS`,
default `gpt-4o-mini=openai/gpt-4o-mini`; a model it doesn't list never fails over).

- **A breaker per provider, per replica:** `GATEWAY_BREAKER_FAILURES` (5) failures
  in a row open it, and after `GATEWAY_BREAKER_COOLDOWN` (30 s) one call goes
  through as a probe (half-open): success closes it, failure opens it again. While
  open, calls skip that provider.
- **What counts as a failure:** 5xx, connection errors, timeouts and the
  provider's own 429s. Never another 4xx: a 400 is the request's fault, and is
  passed on as it came.
- **Timeouts are to the first byte, per endpoint:** a plain chat call's first byte
  is its whole answer (figure extraction runs long, `GATEWAY_CHAT_TIMEOUT` 180 s);
  a stream's is the first byte of its body, not just its headers
  (`GATEWAY_STREAM_TTFB_TIMEOUT` 30 s); embeddings `GATEWAY_EMBEDDINGS_TIMEOUT` 60 s.
- **Never a switch after the first byte has gone back to the caller.** A stream
  that breaks off mid-answer stays broken; the app sees it fail.
- **Embeddings fail over only to an endpoint serving the same
  `text-embedding-3-small`** (e.g. `text-embedding-3-small=openai/text-embedding-3-small`
  in `GATEWAY_BACKUP_MODELS`; the gateway refuses to start with another model
  there). By default there is none, so with OpenAI down embeddings get 503:
  another model's vectors can't be compared with the stored index.
- **Structured output from the backup is checked.** A backup asked for
  `response_format: json_schema` is asked for its answer whole, even when the
  caller streams, and the answer goes out only if it is JSON matching the schema
  (replayed as SSE for a streamed call). Otherwise, or with
  `GATEWAY_BACKUP_JSON_SCHEMA=false`, the call gets 503: never a malformed answer.
- With nowhere else to go, a dropped connection or a 502 is retried once; a
  provider 429 or a timeout is not.
- When no provider can take a call: `503`, `error.code` `providers_unavailable`,
  `x-should-retry: false`. The app degrades where it has a fallback (the
  planner, reranker and follow-up rewrite) and otherwise tells the user.

`X-Gateway-Provider` names who answered (`openai`, `openrouter`, or `cache`), as
do `gateway_served_total{provider}` and `gateway_failovers_total`;
`gateway_breaker_state{provider}` is 0 closed, 1 half-open, 2 open.

`X-Force-Provider: openrouter` sends a call to that provider only, with no
failover and nothing stored in the cache. Only callers with the internal key get
that far; the benchmarks use it to score the backup on its own.

## PII redaction

Before a call is cached or sent anywhere, personal data in its messages (plain
content and text parts) and in embeddings input is replaced with typed
placeholders. High-precision detectors only, each needing the value's own format
and, where one exists, its checksum:

| Placeholder | Found |
| --- | --- |
| `[EMAIL]` | `name@domain.tld` |
| `[PHONE]` | formatted numbers only: `(425) 882-8080`, `425-882-8080`, `425.882.8080`, `1-800-…`, and with a country code `+44 20 7946 0958` |
| `[CARD]` | 13–19 digits, plain or in 4-4-4-4 / 4-6-5 groups, passing the Luhn check |
| `[SSN]` | dashed `123-45-6789`, with a valid area, group and serial |
| `[PAN]` | India's Permanent Account Number, `ABCPE1234F` |
| `[AADHAAR]` | 12 digits, plain or 4-4-4, passing the Verhoeff check |

A digit format is only taken whole, never as a slice of a longer run of digits,
and a run whose groups are all years is never a card or Aadhaar number, so a row
of years in a table ("2024 2023 2022 2021") stays as it is. The test that guards
this runs the redactor over the text of both sample 10-Ks, as the app extracts
it, and every benchmark question: no figure, year, EPS value, share count,
percentage, page or citation may change, no question may change at all, and the
only matches allowed are the companies' own published phone numbers on their
cover pages (`internal/redact/testdata`; `make_fixtures.py` regenerates it).

`gateway_redactions_total{type}` counts what was replaced. The values themselves
are never logged, stored or put in a metric. `GATEWAY_REDACT` picks the kinds
(`all` by default, `off`, or e.g. `email,phone`).

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `LLM_GATEWAY_KEY` | (required) | the key callers present, 16+ characters |
| `OPENAI_API_KEY` | (required) | |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | |
| `GATEWAY_ADDR`, `GATEWAY_METRICS_ADDR` | `:8080`, `:9090` | |
| `GATEWAY_CHAT_TIMEOUT` | `180s` | time to the first byte of a plain chat call, which is the whole answer (figure extraction runs long) |
| `GATEWAY_STREAM_TTFB_TIMEOUT` | `30s` | time to a stream's first byte; the rest may take as long as it takes |
| `GATEWAY_EMBEDDINGS_TIMEOUT` | `60s` | |
| `GATEWAY_MAX_BODY_BYTES` | `16777216` | |
| `GATEWAY_LIMITS_REDIS_URL` | (unset: limits per process) | `redis://llm-gateway-limits:6379/0` in compose |
| `GATEWAY_RATE_PER_CONSUMER`, `GATEWAY_BURST_PER_CONSUMER` | `60` a minute, `20` | per `user:` consumer; `0` turns it off |
| `GATEWAY_RATE_GLOBAL`, `GATEWAY_BURST_GLOBAL` | `300` a minute, `60` | all consumers together |
| `GATEWAY_DAILY_BUDGET_USER` | `320000` | weighted tokens per UTC day, per user; `0` turns it off |
| `GATEWAY_DAILY_BUDGET_SYSTEM` | `0` (none) | per `system:` consumer |
| `GATEWAY_COMPLETION_ESTIMATE` | `512` | answer tokens reserved for a call without `max_tokens` |
| `GATEWAY_PRICES` | | JSON, e.g. `{"gpt-5-nano": {"input": 0.05, "cached": 0.005, "output": 0.4}}` (USD per million) |
| `OPENROUTER_API_KEY` | (unset: no backup) | |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | |
| `GATEWAY_BACKUP_MODELS` | `gpt-4o-mini=openai/gpt-4o-mini` | `asked=backup,…`; embeddings only to the same model |
| `GATEWAY_BACKUP_JSON_SCHEMA` | `true` | the backup can follow `json_schema`; `false` makes such calls 503 instead |
| `GATEWAY_BREAKER_FAILURES`, `GATEWAY_BREAKER_COOLDOWN` | `5`, `30s` | |
| `GATEWAY_REDACT` | `all` | `off`, or a list: `email,phone,card,ssn,pan,aadhaar` |
| `GATEWAY_CACHE_REDIS_URL` | (unset: no cache) | `redis://llm-gateway-cache:6379/0` in compose |
| `GATEWAY_CACHE_TTL` | `168h` | how long an answer is kept |
| `GATEWAY_REPLICAS` | `3` | how many replicas share the limits; the fallback divides by it |

## Development

```bash
cd gateway
go vet ./... && go test -race ./...
GATEWAY_TEST_REDIS_URL=redis://localhost:6379/0 go test -run RealRedis ./internal/limits/   # against a real Redis
go run ./cmd/fakeupstream -addr 127.0.0.1:9100     # a fake OpenAI, free and offline
OPENAI_BASE_URL=http://127.0.0.1:9100/v1 OPENAI_API_KEY=sk-fake \
  LLM_GATEWAY_KEY=local-key-0123456789 go run ./cmd/gateway
```

`RUN_GATEWAY_TESTS=1 pytest tests/test_gateway_live.py` runs the app's real
langchain models (invoke, stream, structured output, embeddings) through the
gateway and the fake upstream; it builds both with Go.

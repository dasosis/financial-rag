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

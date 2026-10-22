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

## Development

```bash
cd gateway
go vet ./... && go test -race ./...
go run ./cmd/fakeupstream -addr 127.0.0.1:9100     # a fake OpenAI, free and offline
OPENAI_BASE_URL=http://127.0.0.1:9100/v1 OPENAI_API_KEY=sk-fake \
  LLM_GATEWAY_KEY=local-key-0123456789 go run ./cmd/gateway
```

`RUN_GATEWAY_TESTS=1 pytest tests/test_gateway_live.py` runs the app's real
langchain models (invoke, stream, structured output, embeddings) through the
gateway and the fake upstream; it builds both with Go.

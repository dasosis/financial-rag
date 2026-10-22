package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/dasosis/financial-rag/gateway/internal/config"
	"github.com/dasosis/financial-rag/gateway/internal/fakeupstream"
	"github.com/dasosis/financial-rag/gateway/internal/metrics"
)

const testKey = "internal-test-key-0123456789"

type harness struct {
	t       *testing.T
	fake    *fakeupstream.Fake
	server  *Server
	gateway *httptest.Server
	metrics *metrics.Metrics
}

func testConfig(upstream string) config.Config {
	return config.Config{
		Key: testKey, OpenAIKey: "sk-upstream", OpenAIBaseURL: upstream + "/v1",
		ChatTimeout: 5 * time.Second, StreamTTFBTimeout: 2 * time.Second,
		EmbedTimeout: 5 * time.Second, MaxBodyBytes: 1 << 20,
	}
}

func newHarness(t *testing.T, opts fakeupstream.Options, tweak ...func(*config.Config)) *harness {
	t.Helper()
	fake := fakeupstream.New(opts)
	upstream := httptest.NewServer(fake)
	t.Cleanup(upstream.Close)
	cfg := testConfig(upstream.URL)
	for _, f := range tweak {
		f(&cfg)
	}
	m := metrics.New()
	srv := New(cfg, Deps{Metrics: m, Log: slog.New(slog.NewTextHandler(io.Discard, nil))})
	gw := httptest.NewServer(srv.Handler())
	t.Cleanup(gw.Close)
	return &harness{t: t, fake: fake, server: srv, gateway: gw, metrics: m}
}

func (h *harness) post(path, body string, headers map[string]string) *http.Response {
	h.t.Helper()
	req, _ := http.NewRequest(http.MethodPost, h.gateway.URL+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testKey)
	req.Header.Set("X-Consumer-Id", "user:alice")
	for k, v := range headers {
		if v == "" {
			req.Header.Del(k)
		} else {
			req.Header.Set(k, v)
		}
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

func readJSON(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func errorCode(t *testing.T, resp *http.Response) string {
	t.Helper()
	body := readJSON(t, resp)
	e, _ := body["error"].(map[string]any)
	code, _ := e["code"].(string)
	return code
}

const chatBody = `{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"What was revenue?"}]}`
const streamBody = `{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"What was revenue?"}]}`

func TestCallsWithoutTheKeyOrAConsumerAreRefused(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	cases := []struct {
		name    string
		headers map[string]string
		code    string
	}{
		{"no key", map[string]string{"Authorization": ""}, "unauthorized"},
		{"wrong key", map[string]string{"Authorization": "Bearer nope"}, "unauthorized"},
		{"not bearer", map[string]string{"Authorization": "Basic " + testKey}, "unauthorized"},
		{"no consumer", map[string]string{"X-Consumer-Id": ""}, "missing_consumer"},
		{"unnamed consumer", map[string]string{"X-Consumer-Id": "alice"}, "missing_consumer"},
		{"unknown kind", map[string]string{"X-Consumer-Id": "admin:alice"}, "missing_consumer"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := h.post("/v1/chat/completions", chatBody, tc.headers)
			if resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("status %d, want 401", resp.StatusCode)
			}
			if code := errorCode(t, resp); code != tc.code {
				t.Fatalf("code %q, want %q", code, tc.code)
			}
		})
	}
	if h.fake.Calls() != 0 {
		t.Fatalf("the provider was called %d times", h.fake.Calls())
	}
}

func TestAChatCallIsPassedThrough(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	resp := h.post("/v1/chat/completions", chatBody, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if resp.Header.Get("X-Request-Id") == "" {
		t.Error("no X-Request-Id")
	}
	body := readJSON(t, resp)
	msg := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if !strings.HasPrefix(msg["content"].(string), "Fake answer") {
		t.Fatalf("unexpected answer %v", msg)
	}
	// The provider sees the provider's key, never the gateway's, and not who asked
	heads := h.fake.LastHeaders()
	if got := heads.Get("Authorization"); got != "Bearer sk-upstream" {
		t.Errorf("provider got Authorization %q", got)
	}
	if heads.Get("X-Consumer-Id") != "" {
		t.Error("the consumer id was forwarded to the provider")
	}
	if got := h.fake.Last()["temperature"]; got != float64(0) {
		t.Errorf("temperature not passed through: %v", got)
	}
	if testutil.ToFloat64(h.metrics.Tokens.WithLabelValues("gpt-4o-mini", "prompt")) == 0 {
		t.Error("prompt tokens not counted")
	}
	if testutil.ToFloat64(h.metrics.Requests.WithLabelValues("chat", "200")) != 1 {
		t.Error("request not counted")
	}
}

// sseEvents reads a stream's data payloads, in order.
func sseEvents(t *testing.T, r io.Reader) []string {
	t.Helper()
	var out []string
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		if data, ok := strings.CutPrefix(scanner.Text(), "data: "); ok {
			out = append(out, data)
		}
	}
	return out
}

func usageChunks(events []string) int {
	n := 0
	for _, e := range events {
		var chunk struct {
			Choices []any           `json:"choices"`
			Usage   json.RawMessage `json:"usage"`
		}
		if json.Unmarshal([]byte(e), &chunk) == nil && len(chunk.Usage) > 0 &&
			string(chunk.Usage) != "null" {
			n++
		}
	}
	return n
}

func TestAStreamAsksForUsageAndDropsItForCallersWhoDidnt(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	resp := h.post("/v1/chat/completions", streamBody, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type %q", ct)
	}
	events := sseEvents(t, resp.Body)
	so, _ := h.fake.Last()["stream_options"].(map[string]any)
	if so["include_usage"] != true {
		t.Fatalf("the provider wasn't asked for usage: %v", h.fake.Last()["stream_options"])
	}
	if n := usageChunks(events); n != 0 {
		t.Fatalf("%d usage chunks reached a caller who didn't ask", n)
	}
	if events[len(events)-1] != "[DONE]" {
		t.Fatalf("stream ended with %q", events[len(events)-1])
	}
	var text strings.Builder
	for _, e := range events[:len(events)-1] {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		_ = json.Unmarshal([]byte(e), &chunk)
		for _, c := range chunk.Choices {
			text.WriteString(c.Delta.Content)
		}
	}
	if !strings.HasPrefix(text.String(), "Fake answer") {
		t.Fatalf("streamed text %q", text.String())
	}
	if testutil.ToFloat64(h.metrics.Tokens.WithLabelValues("gpt-4o-mini", "completion")) == 0 {
		t.Error("completion tokens not counted from the usage chunk")
	}
}

func TestAStreamKeepsUsageForCallersWhoAskedForIt(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	body := `{"model":"gpt-4o-mini","stream":true,"stream_options":{"include_usage":true},` +
		`"messages":[{"role":"user","content":"hi"}]}`
	resp := h.post("/v1/chat/completions", body, nil)
	if n := usageChunks(sseEvents(t, resp.Body)); n != 1 {
		t.Fatalf("%d usage chunks, want 1", n)
	}
}

func TestAStreamIsNotBuffered(t *testing.T) {
	// Twelve-odd words, 100 ms apart: the first must arrive long before the last.
	h := newHarness(t, fakeupstream.Options{ChunkDelay: 100 * time.Millisecond})
	start := time.Now()
	resp := h.post("/v1/chat/completions", streamBody, nil)
	reader := bufio.NewReader(resp.Body)
	var first time.Duration
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		if first == 0 && strings.Contains(line, `"content":"Fake`) {
			first = time.Since(start)
		}
	}
	total := time.Since(start)
	if first == 0 {
		t.Fatal("no content chunk")
	}
	if total < 500*time.Millisecond || first > total/3 {
		t.Fatalf("first content after %v of %v: buffered", first, total)
	}
}

func TestProviderErrorsArePassedOn(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	h.fake.Fail(http.StatusBadRequest, "context_length_exceeded")
	resp := h.post("/v1/chat/completions", chatBody, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != "context_length_exceeded" {
		t.Fatalf("code %q", code)
	}
}

func TestAProviderThatIsDownIsABadGateway(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{}, func(c *config.Config) {
		c.OpenAIBaseURL = "http://127.0.0.1:1/v1" // nothing listens there
	})
	resp := h.post("/v1/chat/completions", chatBody, nil)
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status %d", resp.StatusCode)
	}
	// Retryable, as a failed call to OpenAI itself would be
	if resp.Header.Get("x-should-retry") != "" {
		t.Fatal("x-should-retry set on a retryable failure")
	}
}

func TestAStreamWithNoFirstByteInTimeIsAGatewayTimeout(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{Latency: time.Second}, func(c *config.Config) {
		c.StreamTTFBTimeout = 100 * time.Millisecond
	})
	start := time.Now()
	resp := h.post("/v1/chat/completions", streamBody, nil)
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if elapsed := time.Since(start); elapsed > 800*time.Millisecond {
		t.Fatalf("took %v: the timeout wasn't applied", elapsed)
	}
}

func TestASlowStreamIsNotCutOffAfterItsFirstByte(t *testing.T) {
	// The first-byte timeout is shorter than the whole stream, which must still finish
	h := newHarness(t, fakeupstream.Options{ChunkDelay: 50 * time.Millisecond}, func(c *config.Config) {
		c.StreamTTFBTimeout = 200 * time.Millisecond
	})
	events := sseEvents(t, h.post("/v1/chat/completions", streamBody, nil).Body)
	if events[len(events)-1] != "[DONE]" {
		t.Fatal("the stream was cut off")
	}
}

func TestEmbeddingsArePassedThrough(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{Dimensions: 8})
	resp := h.post("/v1/embeddings",
		`{"model":"text-embedding-3-small","input":["Total revenue","Net income"]}`, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	body := readJSON(t, resp)
	if n := len(body["data"].([]any)); n != 2 {
		t.Fatalf("%d embeddings", n)
	}
	if testutil.ToFloat64(h.metrics.Tokens.WithLabelValues("text-embedding-3-small", "prompt")) == 0 {
		t.Error("embedding tokens not counted")
	}
}

func TestMalformedAndOversizedBodiesAreRefused(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	if resp := h.post("/v1/chat/completions", `{"messages":[]}`, nil); resp.StatusCode != 400 {
		t.Errorf("no model: status %d", resp.StatusCode)
	}
	if resp := h.post("/v1/chat/completions", `not json`, nil); resp.StatusCode != 400 {
		t.Errorf("not JSON: status %d", resp.StatusCode)
	}
	big := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", 2<<20) + `"}]}`
	if resp := h.post("/v1/chat/completions", big, nil); resp.StatusCode != 413 {
		t.Errorf("too large: status %d", resp.StatusCode)
	}
	if h.fake.Calls() != 0 {
		t.Error("the provider was called")
	}
}

func TestHealthAndReadiness(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	get := func(path string) int {
		resp, err := http.Get(h.gateway.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
		return resp.StatusCode
	}
	if get("/healthz") != 200 || get("/readyz") != 200 {
		t.Fatal("not healthy and ready")
	}
	h.server.SetReady(false)
	if get("/readyz") != 503 || get("/healthz") != 200 {
		t.Fatal("a server shutting down should be healthy but not ready")
	}
}

func TestInspectChunk(t *testing.T) {
	var r streamResult
	if !inspectChunk([]byte(`{"choices":[{"delta":{"content":"a"},"finish_reason":null}],"usage":null}`), &r, false) {
		t.Error("content chunk dropped")
	}
	if !inspectChunk([]byte(`{"choices":[{"delta":{},"finish_reason":"stop"}]}`), &r, false) || r.finish != "stop" {
		t.Error("finish reason not seen")
	}
	if inspectChunk([]byte(`{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}`), &r, false) {
		t.Error("usage chunk kept for a caller who didn't ask")
	}
	if r.usage.PromptTokens != 3 || r.usage.CompletionTokens != 2 {
		t.Errorf("usage %+v", r.usage)
	}
	if !inspectChunk([]byte(`[DONE]`), &r, false) || !r.done {
		t.Error("[DONE] not seen")
	}
	if !r.complete() {
		t.Error("a finished stream should be complete")
	}
}

func TestReadEventSplitsOnBlankLines(t *testing.T) {
	r := bufio.NewReader(bytes.NewBufferString("data: a\n\n: comment\n\ndata: b\r\n\r\n"))
	var events []string
	for {
		e, err := readEvent(r)
		if len(e) > 0 {
			events = append(events, string(e))
		}
		if err != nil {
			break
		}
	}
	if len(events) != 3 || events[2] != "data: b\r\n\r\n" {
		t.Fatalf("events %q", events)
	}
	if d, ok := eventData([]byte(events[2])); !ok || string(d) != "b" {
		t.Fatalf("data %q", d)
	}
}

package server

import (
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

// pair is a gateway with OpenAI and the backup, each a fake.
type pair struct {
	*harness
	primary, backup *fakeupstream.Fake
}

func newPair(t *testing.T, primary, backup fakeupstream.Options, tweak ...func(*config.Config)) *pair {
	t.Helper()
	p, b := fakeupstream.New(primary), fakeupstream.New(backup)
	ps, bs := httptest.NewServer(p), httptest.NewServer(b)
	t.Cleanup(ps.Close)
	t.Cleanup(bs.Close)
	cfg := testConfig(ps.URL)
	cfg.OpenRouterKey, cfg.OpenRouterBaseURL = "sk-or-backup", bs.URL+"/v1"
	for _, f := range tweak {
		f(&cfg)
	}
	m := metrics.New()
	srv := New(cfg, Deps{Metrics: m, Log: slog.New(slog.NewTextHandler(io.Discard, nil))})
	gw := httptest.NewServer(srv.Handler())
	t.Cleanup(gw.Close)
	return &pair{harness: &harness{t: t, fake: p, server: srv, gateway: gw, metrics: m}, primary: p, backup: b}
}

func (p *pair) breakerState(provider string) float64 {
	return testutil.ToFloat64(p.metrics.Breaker.WithLabelValues(provider))
}

func TestAFailingPrimaryFailsOverToTheBackup(t *testing.T) {
	for _, status := range []int{500, 502, 503, 429} {
		p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{})
		p.primary.Fail(status, "trouble")
		resp := p.post("/v1/chat/completions", chatBody, nil)
		if resp.StatusCode != 200 || resp.Header.Get("X-Gateway-Provider") != "openrouter" {
			t.Fatalf("primary %d: status %d from %q", status, resp.StatusCode, resp.Header.Get("X-Gateway-Provider"))
		}
		// The backup gets its own key and its own name for the model
		if got := p.backup.LastHeaders().Get("Authorization"); got != "Bearer sk-or-backup" {
			t.Fatalf("backup got Authorization %q", got)
		}
		if model := p.backup.Last()["model"]; model != "openai/gpt-4o-mini" {
			t.Fatalf("backup asked for %v", model)
		}
		if testutil.ToFloat64(p.metrics.Failovers.WithLabelValues("chat")) != 1 ||
			testutil.ToFloat64(p.metrics.Served.WithLabelValues("openrouter", "chat")) != 1 {
			t.Fatal("failover not counted")
		}
	}
}

func TestOtherClientErrorsArePassedOnWithoutFailingOver(t *testing.T) {
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{})
	p.primary.Fail(http.StatusBadRequest, "context_length_exceeded")
	for range 10 {
		resp := p.post("/v1/chat/completions", chatBody, nil)
		if resp.StatusCode != 400 || resp.Header.Get("X-Gateway-Provider") != "openai" {
			t.Fatalf("status %d from %q", resp.StatusCode, resp.Header.Get("X-Gateway-Provider"))
		}
	}
	if p.backup.Calls() != 0 || p.breakerState("openai") != 0 {
		t.Fatal("a 400 counted against the provider")
	}
}

func TestTheBreakerOpensThenProbes(t *testing.T) {
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{}, func(c *config.Config) {
		c.BreakerFailures, c.BreakerCooldown = 3, 300*time.Millisecond
	})
	p.primary.Fail(500, "server_error")
	for range 6 {
		if resp := p.post("/v1/chat/completions", chatBody, nil); resp.StatusCode != 200 {
			t.Fatalf("status %d", resp.StatusCode)
		}
	}
	if calls := p.primary.Calls(); calls != 3 {
		t.Fatalf("the primary was called %d times; its breaker should open after 3", calls)
	}
	if p.breakerState("openai") != 2 {
		t.Fatalf("breaker state %v, want open", p.breakerState("openai"))
	}
	time.Sleep(350 * time.Millisecond)
	p.primary.Fail(0, "") // OpenAI is back
	resp := p.post("/v1/chat/completions", chatBody, nil)
	if resp.Header.Get("X-Gateway-Provider") != "openai" || p.breakerState("openai") != 0 {
		t.Fatal("the probe after the cooldown didn't close the breaker")
	}
}

func TestAStallingStreamFailsOverBeforeItsFirstByte(t *testing.T) {
	// Headers at once, then nothing: the first byte, not the headers, decides
	p := newPair(t, fakeupstream.Options{StallBody: 2 * time.Second}, fakeupstream.Options{},
		func(c *config.Config) { c.StreamTTFBTimeout = 150 * time.Millisecond })
	start := time.Now()
	resp := p.post("/v1/chat/completions", streamBody, nil)
	events := sseEvents(t, resp.Body)
	if resp.Header.Get("X-Gateway-Provider") != "openrouter" || events[len(events)-1] != "[DONE]" {
		t.Fatalf("from %q: %v", resp.Header.Get("X-Gateway-Provider"), events)
	}
	if time.Since(start) > time.Second {
		t.Fatal("waited for the stalled stream")
	}
}

func TestNoSwitchAfterTheFirstByte(t *testing.T) {
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{})
	p.primary.Truncate(true) // streams half the answer, then the connection drops
	resp := p.post("/v1/chat/completions", streamBody, nil)
	events := sseEvents(t, resp.Body)
	if resp.Header.Get("X-Gateway-Provider") != "openai" || p.backup.Calls() != 0 {
		t.Fatal("switched providers mid-stream")
	}
	if len(events) == 0 || events[len(events)-1] == "[DONE]" {
		t.Fatal("the cut-off stream was completed from somewhere")
	}
}

func TestEmbeddingsFailOverOnlyToTheSameModel(t *testing.T) {
	embed := `{"model":"text-embedding-3-small","input":["Total revenue"]}`
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{Dimensions: 4}) // backup: chat only
	p.primary.Fail(503, "unavailable")
	resp := p.post("/v1/embeddings", embed, nil)
	if resp.StatusCode != 503 || errorCode(t, resp) != CodeProvidersUnavailable || p.backup.Calls() != 0 {
		t.Fatalf("status %d; the backup got %d calls", resp.StatusCode, p.backup.Calls())
	}

	same := newPair(t, fakeupstream.Options{}, fakeupstream.Options{Dimensions: 4}, func(c *config.Config) {
		c.BackupModels = map[string]string{"text-embedding-3-small": "openai/text-embedding-3-small"}
	})
	same.primary.Fail(503, "unavailable")
	resp = same.post("/v1/embeddings", embed, nil)
	if resp.StatusCode != 200 || resp.Header.Get("X-Gateway-Provider") != "openrouter" ||
		same.backup.Last()["model"] != "openai/text-embedding-3-small" {
		t.Fatalf("status %d, model %v", resp.StatusCode, same.backup.Last()["model"])
	}

	other := newPair(t, fakeupstream.Options{}, fakeupstream.Options{Dimensions: 4}, func(c *config.Config) {
		c.BackupModels = map[string]string{"text-embedding-3-small": "text-embedding-3-large"}
	})
	other.primary.Fail(503, "unavailable")
	if resp := other.post("/v1/embeddings", embed, nil); resp.StatusCode != 503 || other.backup.Calls() != 0 {
		t.Fatal("embeddings failed over to another model")
	}
}

func TestAModelTheBackupDoesntServeDoesntFailOver(t *testing.T) {
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{})
	p.primary.Fail(500, "server_error")
	resp := p.post("/v1/chat/completions", strings.Replace(chatBody, "gpt-4o-mini", "gpt-5-mini", 1), nil)
	if resp.StatusCode != 503 || p.backup.Calls() != 0 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestXForceProviderSendsTheCallThereOnly(t *testing.T) {
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{})
	resp := p.post("/v1/chat/completions", chatBody, map[string]string{"X-Force-Provider": "openrouter"})
	if resp.Header.Get("X-Gateway-Provider") != "openrouter" || p.primary.Calls() != 0 {
		t.Fatal("not sent to the forced provider")
	}
	p.backup.Fail(500, "server_error")
	resp = p.post("/v1/chat/completions", chatBody, map[string]string{"X-Force-Provider": "openrouter"})
	if resp.StatusCode != 503 || p.primary.Calls() != 0 {
		t.Fatal("a forced call failed over")
	}
	resp = p.post("/v1/chat/completions", chatBody, map[string]string{"X-Force-Provider": "mystery"})
	if resp.StatusCode != 400 || errorCode(t, resp) != "unknown_provider" {
		t.Fatal("an unknown provider was accepted")
	}
	// The key still comes first: without it, the header gets nothing
	resp = p.post("/v1/chat/completions", chatBody, map[string]string{"X-Force-Provider": "openrouter",
		"Authorization": "Bearer wrong"})
	if resp.StatusCode != 401 {
		t.Fatal("forced without the key")
	}
}

const structuredBody = `{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"Plan"}],` +
	`"response_format":{"type":"json_schema","json_schema":{"name":"Plan","strict":true,"schema":` +
	`{"type":"object","properties":{"route":{"type":"string","enum":["calculation","text"]},` +
	`"metrics":{"type":"array","items":{"type":"string"}}},"required":["route","metrics"],` +
	`"additionalProperties":false}}}}`

func TestTheBackupsStructuredOutputIsCheckedBeforeItGoesOut(t *testing.T) {
	p := newPair(t, fakeupstream.Options{}, fakeupstream.Options{})
	p.primary.Fail(500, "server_error")
	resp := p.post("/v1/chat/completions", structuredBody, nil)
	if resp.StatusCode != 200 || resp.Header.Get("X-Gateway-Provider") != "openrouter" {
		t.Fatalf("a valid structured answer from the backup: status %d", resp.StatusCode)
	}
	// Streamed: asked whole from the backup, checked, then replayed as SSE
	streamed := strings.Replace(structuredBody, `"temperature":0`, `"temperature":0,"stream":true`, 1)
	resp = p.post("/v1/chat/completions", streamed, nil)
	events := sseEvents(t, resp.Body)
	if resp.Header.Get("Content-Type") != "text/event-stream" || events[len(events)-1] != "[DONE]" {
		t.Fatal("a streamed structured call wasn't answered as a stream")
	}
	if stream, _ := p.backup.Last()["stream"].(bool); stream {
		t.Fatal("the backup was asked to stream a structured answer it must check first")
	}

	bad := newPair(t, fakeupstream.Options{}, fakeupstream.Options{BadJSON: true})
	bad.primary.Fail(500, "server_error")
	resp = bad.post("/v1/chat/completions", structuredBody, nil)
	if resp.StatusCode != 503 || errorCode(t, resp) != CodeProvidersUnavailable {
		t.Fatalf("a malformed structured answer went out: status %d", resp.StatusCode)
	}

	unable := newPair(t, fakeupstream.Options{}, fakeupstream.Options{},
		func(c *config.Config) { c.BackupJSONSchema = false })
	unable.primary.Fail(500, "server_error")
	if resp := unable.post("/v1/chat/completions", structuredBody, nil); resp.StatusCode != 503 || unable.backup.Calls() != 0 {
		t.Fatal("a backup without json_schema was asked for it")
	}
	// Plain calls still fail over to it
	if resp := unable.post("/v1/chat/completions", chatBody, nil); resp.StatusCode != 200 {
		t.Fatal("a plain call didn't fail over")
	}
}

func TestWithOnlyOneProviderABlipIsRetriedOnce(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	h.fake.FailNext(1, 502, "bad_gateway")
	if resp := h.post("/v1/chat/completions", chatBody, nil); resp.StatusCode != 200 || h.fake.Calls() != 2 {
		t.Fatalf("status %d after %d calls", resp.StatusCode, h.fake.Calls())
	}
	h.fake.FailNext(1, 429, "rate_limit_exceeded") // the provider's own limit: no hammering it
	if resp := h.post("/v1/chat/completions", chatBody, nil); resp.StatusCode != 503 || h.fake.Calls() != 3 {
		t.Fatalf("a provider 429 was retried: status %d after %d calls", resp.StatusCode, h.fake.Calls())
	}
}

func TestStructuredCheck(t *testing.T) {
	schema := []byte(`{"type":"object","properties":{"route":{"type":"string","enum":["a","b"]},` +
		`"years":{"type":"array","items":{"type":"integer"}},"q":{"anyOf":[{"type":"string"},{"type":"null"}]},` +
		`"part":{"$ref":"#/$defs/Part"}},"required":["route","years","q","part"],"additionalProperties":false,` +
		`"$defs":{"Part":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}`)
	answer := func(content string) []byte {
		return []byte(`{"choices":[{"message":{"content":` + content + `},"finish_reason":"stop"}]}`)
	}
	ok := map[string]string{
		"valid":   `"{\"route\":\"a\",\"years\":[2024],\"q\":null,\"part\":{\"name\":\"x\"}}"`,
		"refusal": `null,"refusal":"I can't"`,
	}
	for name, content := range ok {
		if err := checkStructured(answer(content), schema); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	bad := map[string]string{
		"not JSON":           `"{\"route\":"`,
		"missing a property": `"{\"route\":\"a\",\"years\":[],\"q\":null}"`,
		"not in the enum":    `"{\"route\":\"c\",\"years\":[],\"q\":null,\"part\":{\"name\":\"x\"}}"`,
		"wrong item type":    `"{\"route\":\"a\",\"years\":[\"2024\"],\"q\":null,\"part\":{\"name\":\"x\"}}"`,
		"an extra property":  `"{\"route\":\"a\",\"years\":[],\"q\":null,\"part\":{\"name\":\"x\"},\"x\":1}"`,
		"a bad $ref value":   `"{\"route\":\"a\",\"years\":[],\"q\":null,\"part\":{}}"`,
		"no content":         `null`,
	}
	for name, content := range bad {
		if err := checkStructured(answer(content), schema); err == nil {
			t.Errorf("%s: passed", name)
		}
	}
	if checkStructured([]byte(`{"choices":[{"message":{"content":"{}"},"finish_reason":"length"}]}`), []byte(`{}`)) == nil {
		t.Error("a truncated answer passed")
	}
}

func TestBackupModelsConfig(t *testing.T) {
	t.Setenv("LLM_GATEWAY_KEY", testKey)
	t.Setenv("OPENAI_API_KEY", "sk")
	t.Setenv("GATEWAY_BACKUP_MODELS", "gpt-4o-mini=openai/gpt-4o-mini, text-embedding-3-small=openai/text-embedding-3-small")
	if cfg, err := config.FromEnv(); err != nil || cfg.BackupModels["text-embedding-3-small"] != "openai/text-embedding-3-small" {
		t.Fatalf("%v %v", cfg.BackupModels, err)
	}
	t.Setenv("GATEWAY_BACKUP_MODELS", "text-embedding-3-small=text-embedding-3-large")
	if _, err := config.FromEnv(); err == nil {
		t.Fatal("an embedding model mapped to another model was accepted")
	}
}

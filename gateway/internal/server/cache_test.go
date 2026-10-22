package server

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"

	"github.com/dasosis/financial-rag/gateway/internal/cache"
	"github.com/dasosis/financial-rag/gateway/internal/fakeupstream"
	"github.com/dasosis/financial-rag/gateway/internal/limits"
	"github.com/dasosis/financial-rag/gateway/internal/metrics"
)

type cached struct {
	*harness
	redis *miniredis.Miniredis
}

func newCached(t *testing.T, opts fakeupstream.Options) *cached {
	t.Helper()
	fake := fakeupstream.New(opts)
	upstream := httptest.NewServer(fake)
	t.Cleanup(upstream.Close)
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr(), ReadTimeout: 10 * time.Second})
	t.Cleanup(func() { _ = rdb.Close() })
	m := metrics.New()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	srv := New(testConfig(upstream.URL), Deps{Metrics: m, Log: log, Cache: cache.New(rdb, 7*24*time.Hour),
		Limiter: limits.New(nil, limits.Settings{UserBudget: 1_000_000}, m, log)})
	gw := httptest.NewServer(srv.Handler())
	t.Cleanup(gw.Close)
	return &cached{harness: &harness{t: t, fake: fake, server: srv, gateway: gw, metrics: m}, redis: mr}
}

func body(t *testing.T, resp *http.Response) string {
	t.Helper()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func (c *cached) count(result string) float64 {
	return testutil.ToFloat64(c.metrics.Cache.WithLabelValues(result))
}

func TestARepeatedCallIsAnsweredFromTheCache(t *testing.T) {
	c := newCached(t, fakeupstream.Options{})
	first := c.post("/v1/chat/completions", chatBody, nil)
	firstBody := body(t, first)
	if first.Header.Get("X-Gateway-Cache") != "miss" {
		t.Fatalf("first call: %q", first.Header.Get("X-Gateway-Cache"))
	}
	second := c.post("/v1/chat/completions", chatBody, nil)
	if second.Header.Get("X-Gateway-Cache") != "hit" || body(t, second) != firstBody {
		t.Fatal("the repeat wasn't the stored answer")
	}
	if c.fake.Calls() != 1 {
		t.Fatalf("the provider was called %d times", c.fake.Calls())
	}
	if c.count("hit") != 1 || c.count("miss") != 1 {
		t.Fatalf("hits %v, misses %v", c.count("hit"), c.count("miss"))
	}
	if testutil.ToFloat64(c.metrics.CacheSaved.WithLabelValues("gpt-4o-mini")) <= 0 {
		t.Fatal("cost saved not counted")
	}
	// Only the response is stored: the key is a hash, the value the answer
	for _, k := range c.redis.Keys() {
		if v, _ := c.redis.Get(k); strings.Contains(v, "What was revenue?") || strings.Contains(k, "revenue") {
			t.Fatal("the prompt was stored")
		}
	}
	if ttl := c.redis.TTL(c.redis.Keys()[0]); ttl != 7*24*time.Hour {
		t.Fatalf("TTL %v", ttl)
	}
}

func TestAnotherYearIsNotTheSameQuestion(t *testing.T) {
	c := newCached(t, fakeupstream.Options{})
	ask := func(year string) string {
		b := `{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"Net income in fiscal ` + year + `?"}]}`
		return body(t, c.post("/v1/chat/completions", b, nil))
	}
	if ask("2023") == ask("2024") || c.fake.Calls() != 2 {
		t.Fatal("2024 was answered with 2023's answer")
	}
}

func TestStreamsAndPlainCallsShareEntries(t *testing.T) {
	c := newCached(t, fakeupstream.Options{})
	streamed := `{"model":"gpt-4o-mini","temperature":0,"stream":true,"messages":[{"role":"user","content":"What was revenue?"}]}`
	events := sseEvents(t, c.post("/v1/chat/completions", streamed, nil).Body) // miss: from the provider
	plain := c.post("/v1/chat/completions", chatBody, nil)                     // hit: the assembled stream
	if plain.Header.Get("X-Gateway-Cache") != "hit" || !strings.Contains(body(t, plain), `"content":"Fake answer`) {
		t.Fatal("a plain call didn't hit the streamed answer")
	}
	again := c.post("/v1/chat/completions", streamed, nil) // hit, replayed as SSE
	if again.Header.Get("Content-Type") != "text/event-stream" || again.Header.Get("X-Gateway-Cache") != "hit" {
		t.Fatal("a streamed hit wasn't SSE")
	}
	replayed := sseEvents(t, again.Body)
	if replayed[len(replayed)-1] != "[DONE]" || usageChunks(replayed) != 0 {
		t.Fatalf("replay %q", replayed)
	}
	if text(events) != text(replayed) {
		t.Fatalf("replayed %q, streamed %q", text(replayed), text(events))
	}
	withUsage := strings.Replace(streamed, `"stream":true`, `"stream":true,"stream_options":{"include_usage":true}`, 1)
	if usageChunks(sseEvents(t, c.post("/v1/chat/completions", withUsage, nil).Body)) != 1 {
		t.Fatal("a caller asking for usage didn't get it from the cache")
	}
	if c.fake.Calls() != 1 {
		t.Fatalf("the provider was called %d times", c.fake.Calls())
	}
}

func text(events []string) string {
	var out strings.Builder
	for _, e := range events {
		if i := strings.Index(e, `"content":"`); i >= 0 {
			rest := e[i+len(`"content":"`):]
			out.WriteString(rest[:strings.Index(rest, `"`)])
		}
	}
	return out.String()
}

func TestNoCacheBypassesTheCache(t *testing.T) {
	c := newCached(t, fakeupstream.Options{})
	c.post("/v1/chat/completions", chatBody, nil)
	resp := c.post("/v1/chat/completions", chatBody, map[string]string{"Cache-Control": "no-cache"})
	if resp.Header.Get("X-Gateway-Cache") != "bypass" || c.fake.Calls() != 2 {
		t.Fatal("no-cache was answered from the cache")
	}
	if c.count("bypass") != 1 {
		t.Fatal("bypass not counted")
	}
}

func TestErrorsTruncationsAndOtherTemperaturesAreNotCached(t *testing.T) {
	cases := map[string]func(c *cached) string{
		"an error": func(c *cached) string {
			c.fake.Fail(http.StatusInternalServerError, "server_error")
			return chatBody
		},
		"a cut-off stream": func(c *cached) string {
			c.fake.Truncate(true)
			return strings.Replace(chatBody, `"temperature":0`, `"temperature":0,"stream":true`, 1)
		},
		"an answer cut at max_tokens": func(c *cached) string {
			return strings.Replace(chatBody, `"temperature":0`, `"temperature":0,"max_tokens":3`, 1)
		},
		"temperature 0.7": func(c *cached) string {
			return strings.Replace(chatBody, `"temperature":0`, `"temperature":0.7`, 1)
		},
	}
	for name, setup := range cases {
		t.Run(name, func(t *testing.T) {
			c := newCached(t, fakeupstream.Options{})
			b := setup(c)
			_, _ = io.Copy(io.Discard, c.post("/v1/chat/completions", b, nil).Body)
			if keys := c.redis.Keys(); len(keys) != 0 {
				t.Fatalf("stored %v", keys)
			}
		})
	}
}

func TestAHitCostsNothing(t *testing.T) {
	c := newCached(t, fakeupstream.Options{})
	c.server.limiter = limits.New(nil, limits.Settings{UserBudget: 1_000_000, ConsumerRate: 1.0 / 60, ConsumerBurst: 1},
		c.metrics, slog.New(slog.NewTextHandler(io.Discard, nil)))
	c.post("/v1/chat/completions", chatBody, nil) // takes the one call the bucket allows
	for range 3 {
		if resp := c.post("/v1/chat/completions", chatBody, nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("a cache hit was rate-limited: %d", resp.StatusCode)
		}
	}
}

func TestACacheThatIsDownIsAMiss(t *testing.T) {
	c := newCached(t, fakeupstream.Options{})
	c.redis.Close()
	for range 2 {
		if resp := c.post("/v1/chat/completions", chatBody, nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("status %d without the cache", resp.StatusCode)
		}
	}
	if c.fake.Calls() != 2 {
		t.Fatal("calls weren't answered by the provider")
	}
}

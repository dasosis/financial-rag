package server

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"

	"github.com/dasosis/financial-rag/gateway/internal/fakeupstream"
	"github.com/dasosis/financial-rag/gateway/internal/limits"
	"github.com/dasosis/financial-rag/gateway/internal/metrics"
	"github.com/dasosis/financial-rag/gateway/internal/pricing"
	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

// cluster is three gateway replicas sharing one Redis and one fake provider.
type cluster struct {
	fake     *fakeupstream.Fake
	redis    *miniredis.Miniredis
	rdb      *redis.Client
	replicas []*httptest.Server
	metrics  []*metrics.Metrics
}

func newCluster(t *testing.T, s limits.Settings) *cluster {
	t.Helper()
	fake := fakeupstream.New(fakeupstream.Options{})
	upstream := httptest.NewServer(fake)
	t.Cleanup(upstream.Close)
	mr := miniredis.RunT(t)
	mr.SetTime(time.Date(2026, 10, 22, 18, 0, 0, 0, time.UTC)) // frozen: nothing refills
	rdb := redis.NewClient(limits.FailFast(&redis.Options{Addr: mr.Addr()}))
	t.Cleanup(func() { _ = rdb.Close() })
	c := &cluster{fake: fake, redis: mr, rdb: rdb}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	for range 3 {
		m := metrics.New()
		l := limits.New(rdb, s, m, log)
		t.Cleanup(l.Close)
		srv := New(testConfig(upstream.URL), Deps{Metrics: m, Log: log, Limiter: l})
		gw := httptest.NewServer(srv.Handler())
		t.Cleanup(gw.Close)
		c.replicas = append(c.replicas, gw)
		c.metrics = append(c.metrics, m)
	}
	return c
}

func (c *cluster) call(t *testing.T, replica int, consumer, path, body string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, c.replicas[replica%3].URL+path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+testKey)
	req.Header.Set("X-Consumer-Id", consumer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Error(err)
		return nil
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	return resp
}

func TestAcrossThreeReplicasNoMoreThanTheGlobalLimitIsAdmitted(t *testing.T) {
	c := newCluster(t, limits.Settings{GlobalRate: 0.01, GlobalBurst: 40, Replicas: 3})
	var ok, limited atomic.Int64
	var wg sync.WaitGroup
	for client := range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 6 {
				resp := c.call(t, client+i, fmt.Sprintf("user:u%d", client), "/v1/chat/completions", chatBody)
				if resp == nil {
					return
				}
				switch resp.StatusCode {
				case http.StatusOK:
					ok.Add(1)
				case http.StatusTooManyRequests:
					limited.Add(1)
					if resp.Header.Get("x-should-retry") != "false" || resp.Header.Get("Retry-After") == "" {
						t.Error("a limit without x-should-retry: false and Retry-After")
					}
				default:
					t.Errorf("status %d", resp.StatusCode)
				}
			}
		}()
	}
	wg.Wait()
	if ok.Load() != 40 || limited.Load() != 80 {
		t.Fatalf("%d admitted, %d limited of 120; the global limit is 40", ok.Load(), limited.Load())
	}
	if c.fake.Calls() != 40 {
		t.Fatalf("the provider got %d calls", c.fake.Calls())
	}
	var rejected float64
	for _, m := range c.metrics {
		rejected += testutil.ToFloat64(m.Rejections.WithLabelValues("global_rate"))
	}
	if rejected != 80 {
		t.Fatalf("%v rejections counted", rejected)
	}
}

func TestARateLimitedCallSaysSo(t *testing.T) {
	h := newHarness(t, fakeupstream.Options{})
	h.server.limiter = limits.New(nil, limits.Settings{ConsumerRate: 1.0 / 60, ConsumerBurst: 1}, h.metrics,
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	h.post("/v1/chat/completions", chatBody, nil)
	resp := h.post("/v1/chat/completions", chatBody, nil)
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != CodeRateLimited {
		t.Fatalf("code %q", code)
	}
	if after, _ := strconv.Atoi(resp.Header.Get("Retry-After")); after < 59 || after > 60 {
		t.Fatalf("Retry-After %q, want about a minute", resp.Header.Get("Retry-After"))
	}
}

func usedBudget(t *testing.T, c *cluster, consumer string) int64 {
	t.Helper()
	used, err := c.rdb.HGet(context.Background(), "gw:budget:{"+consumer+"}", "used").Int64()
	if err != nil && err != redis.Nil {
		t.Fatal(err)
	}
	return used
}

func TestABudgetIsChargedWhatTheProviderReported(t *testing.T) {
	c := newCluster(t, limits.Settings{UserBudget: 1_000_000, Replicas: 3})
	if resp := c.call(t, 0, "user:alice", "/v1/chat/completions", chatBody); resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	// What the fake reports for this prompt and answer, in budget units
	prompt := len(`[{"role":"user","content":"What was revenue?"}]`)/4 + 1
	answer := fakeupstream.Answer(map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": "What was revenue?"}}})
	prices, _ := pricing.New("")
	want := prices.Units("gpt-4o-mini", usage.Usage{PromptTokens: prompt, CompletionTokens: len(answer)/4 + 1})
	if got := usedBudget(t, c, "user:alice"); got != want {
		t.Fatalf("charged %d units, want %d (the reported usage, not the estimate)", got, want)
	}
	// Streams too, from the usage chunk the gateway asked for
	c.call(t, 1, "user:alice", "/v1/chat/completions", streamBody)
	if got := usedBudget(t, c, "user:alice"); got != 2*want {
		t.Fatalf("after a stream, charged %d units, want %d", got, 2*want)
	}
}

func TestAFailedCallIsRefunded(t *testing.T) {
	c := newCluster(t, limits.Settings{UserBudget: 1_000_000, Replicas: 3})
	c.fake.Fail(http.StatusInternalServerError, "server_error")
	c.call(t, 0, "user:alice", "/v1/chat/completions", chatBody)
	c.call(t, 1, "user:alice", "/v1/embeddings", `{"model":"text-embedding-3-small","input":"x"}`)
	if got := usedBudget(t, c, "user:alice"); got != 0 {
		t.Fatalf("charged %d units for failed calls", got)
	}
}

func TestASpentBudgetIsRefusedUntilMidnightUTC(t *testing.T) {
	// Reservations here are the prompt alone (the test config estimates no
	// answer), so each call is charged more than it reserved: about 70 units
	c := newCluster(t, limits.Settings{UserBudget: 300, Replicas: 3})
	refused := 0
	for i := range 10 {
		resp := c.call(t, i, "user:alice", "/v1/chat/completions", chatBody)
		if resp.StatusCode == http.StatusTooManyRequests {
			refused++
			if after := resp.Header.Get("Retry-After"); after != strconv.Itoa(6*3600) {
				t.Fatalf("Retry-After %s, want 6 h (it's 18:00 UTC)", after)
			}
		}
	}
	if refused == 0 {
		t.Fatal("a 300-unit budget never ran out")
	}
	// Another user, and the system, are unaffected
	if c.call(t, 0, "user:bob", "/v1/chat/completions", chatBody).StatusCode != 200 {
		t.Fatal("another user was refused")
	}
	if c.call(t, 0, "system:ingest", "/v1/chat/completions", chatBody).StatusCode != 200 {
		t.Fatal("system:ingest was refused")
	}
	if status, code := c.callCode(t, "user:alice"); status != 429 || code != CodeBudgetExhausted {
		t.Fatalf("%d %q, want 429 budget_exhausted", status, code)
	}
}

// callCode makes a chat call and returns its status and error code.
func (c *cluster) callCode(t *testing.T, consumer string) (int, string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, c.replicas[0].URL+"/v1/chat/completions", strings.NewReader(chatBody))
	req.Header.Set("Authorization", "Bearer "+testKey)
	req.Header.Set("X-Consumer-Id", consumer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return resp.StatusCode, ""
	}
	return resp.StatusCode, errorCode(t, resp)
}

func TestWithoutRedisTheReplicasStillLimit(t *testing.T) {
	c := newCluster(t, limits.Settings{GlobalRate: 0.01, GlobalBurst: 30, Replicas: 3})
	c.redis.Close()
	ok := 0
	for i := range 60 {
		if c.call(t, i, "user:alice", "/v1/chat/completions", chatBody).StatusCode == 200 {
			ok++
		}
	}
	if ok != 30 { // 10 per replica: a third of 30 each
		t.Fatalf("%d admitted without Redis, want 30", ok)
	}
	for _, m := range c.metrics {
		if testutil.ToFloat64(m.LimiterDegraded) != 1 {
			t.Fatal("limiter_degraded not set")
		}
	}
}

func TestPricesAndUnits(t *testing.T) {
	prices, err := pricing.New(`{"gpt-5-nano": {"input": 0.05, "output": 0.4}}`)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		model string
		u     usage.Usage
		units int64
	}{
		{"gpt-4o-mini", usage.Usage{PromptTokens: 1000}, 1000},
		{"gpt-4o-mini-2024-07-18", usage.Usage{CompletionTokens: 1000}, 4000},
		{"gpt-4o-mini", usage.Usage{PromptTokens: 1000, PromptTokensDetails: &struct {
			CachedTokens int `json:"cached_tokens"`
		}{CachedTokens: 1000}}, 500},
		{"text-embedding-3-small", usage.Usage{PromptTokens: 15000}, 2000},
		{"gpt-4o", usage.Usage{PromptTokens: 1000}, 16667},
		{"gpt-5-nano", usage.Usage{PromptTokens: 3000}, 1000},
		{"some-new-model", usage.Usage{PromptTokens: 1000}, 16667}, // charged as gpt-4o
	}
	for _, tc := range cases {
		if got := prices.Units(tc.model, tc.u); got != tc.units {
			t.Errorf("%s %+v: %d units, want %d", tc.model, tc.u, got, tc.units)
		}
	}
	if _, err := pricing.New("not json"); err == nil {
		t.Error("bad GATEWAY_PRICES accepted")
	}
}

package limits

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

type fakeMetrics struct {
	degraded atomic.Bool
	rejected sync.Map // Reason -> *atomic.Int64
}

func (m *fakeMetrics) SetDegraded(on bool) { m.degraded.Store(on) }
func (m *fakeMetrics) Rejected(r Reason) {
	v, _ := m.rejected.LoadOrStore(r, new(atomic.Int64))
	v.(*atomic.Int64).Add(1)
}

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

func newRedis(t *testing.T) (*miniredis.Miniredis, *redis.Client) {
	t.Helper()
	m := miniredis.RunT(t)
	m.SetTime(time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC))
	rdb := redis.NewClient(FailFast(&redis.Options{Addr: m.Addr()}))
	t.Cleanup(func() { _ = rdb.Close() })
	return m, rdb
}

func newLimiter(t *testing.T, rdb redis.UniversalClient, s Settings) (*Limiter, *fakeMetrics) {
	t.Helper()
	m := &fakeMetrics{}
	l := New(rdb, s, m, quiet)
	t.Cleanup(l.Close)
	return l, m
}

// Three replicas (limiters) on one Redis, and many clients at once: no more
// calls get through than the global bucket holds. Redis's clock is frozen, so
// nothing refills during the test and the bound is exact.
func TestTheGlobalLimitHoldsAcrossReplicas(t *testing.T) {
	_, rdb := newRedis(t)
	s := Settings{GlobalRate: 1, GlobalBurst: 50, ConsumerRate: 100, ConsumerBurst: 100, Replicas: 3}
	var replicas []*Limiter
	for range 3 {
		l, _ := newLimiter(t, rdb, s)
		replicas = append(replicas, l)
	}
	var admitted atomic.Int64
	var wg sync.WaitGroup
	for client := range 30 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 10 {
				l := replicas[(client+i)%3]
				if l.Admit(context.Background(), fmt.Sprintf("user:c%d", client), 0).Allowed {
					admitted.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	if got := admitted.Load(); got != 50 {
		t.Fatalf("%d calls admitted out of 300, want exactly the global burst of 50", got)
	}
	for _, l := range replicas {
		if l.Degraded() {
			t.Fatal("a replica fell back to local limits")
		}
	}
}

func TestAConsumerIsLimitedOnItsOwnAndRefillsByRedisTime(t *testing.T) {
	m, rdb := newRedis(t)
	l, metrics := newLimiter(t, rdb, Settings{ConsumerRate: 0.5, ConsumerBurst: 2})
	ctx := context.Background()
	for i := range 2 {
		if !l.Admit(ctx, "user:alice", 0).Allowed {
			t.Fatalf("call %d refused", i)
		}
	}
	d := l.Admit(ctx, "user:alice", 0)
	if d.Allowed || d.Reason != ConsumerRate {
		t.Fatalf("third call: %+v", d)
	}
	if d.RetryAfter != 2*time.Second {
		t.Fatalf("retry after %v, want 2s at half a call per second", d.RetryAfter)
	}
	if !l.Admit(ctx, "user:bob", 0).Allowed {
		t.Fatal("another consumer was limited too")
	}
	for range 5 { // a benchmark or ingest run: only the global bucket paces it
		if !l.Admit(ctx, "system:eval", 0).Allowed {
			t.Fatal("a system consumer was held to the per-user rate")
		}
	}
	m.SetTime(time.Date(2026, 10, 22, 12, 0, 2, 0, time.UTC)) // Redis's clock, not ours
	if !l.Admit(ctx, "user:alice", 0).Allowed {
		t.Fatal("the bucket didn't refill as Redis's time passed")
	}
	if v, _ := metrics.rejected.Load(ConsumerRate); v.(*atomic.Int64).Load() != 1 {
		t.Fatal("rejection not counted")
	}
}

func TestABudgetIsReservedThenCorrected(t *testing.T) {
	_, rdb := newRedis(t)
	l, _ := newLimiter(t, rdb, Settings{UserBudget: 1000})
	ctx := context.Background()

	d := l.Admit(ctx, "user:alice", 600)
	if !d.Allowed {
		t.Fatal("first call refused")
	}
	if l.Admit(ctx, "user:alice", 600).Allowed {
		t.Fatal("a reservation past the budget was allowed")
	}
	l.Settle(ctx, d.Reservation, 100) // it cost much less than estimated
	if !l.Admit(ctx, "user:alice", 600).Allowed {
		t.Fatal("the correction wasn't applied")
	}
	used, _ := rdb.HGet(ctx, budgetKey("user:alice"), "used").Int64()
	if used != 700 {
		t.Fatalf("used %d, want 100 + 600", used)
	}
}

func TestAFailedCallIsRefundedAndAnUnknownCostKept(t *testing.T) {
	_, rdb := newRedis(t)
	l, _ := newLimiter(t, rdb, Settings{UserBudget: 1000})
	ctx := context.Background()
	l.Settle(ctx, l.Admit(ctx, "user:alice", 400).Reservation, 0)  // failed: refunded
	l.Settle(ctx, l.Admit(ctx, "user:alice", 300).Reservation, -1) // unknown: kept
	used, _ := rdb.HGet(ctx, budgetKey("user:alice"), "used").Int64()
	if used != 300 {
		t.Fatalf("used %d, want 300", used)
	}
}

func TestAnExhaustedBudgetWaitsForMidnightUTCAndThenResets(t *testing.T) {
	m, rdb := newRedis(t)
	l, metrics := newLimiter(t, rdb, Settings{UserBudget: 1000, SystemBudget: 0})
	ctx := context.Background()
	l.Admit(ctx, "user:alice", 1000)
	d := l.Admit(ctx, "user:alice", 1)
	if d.Allowed || d.Reason != Budget {
		t.Fatalf("over budget: %+v", d)
	}
	if d.RetryAfter != 12*time.Hour { // the clock reads 12:00 UTC
		t.Fatalf("retry after %v, want 12h to midnight UTC", d.RetryAfter)
	}
	if v, _ := metrics.rejected.Load(Budget); v == nil {
		t.Fatal("budget rejection not counted")
	}
	// System consumers have no budget by default
	if !l.Admit(ctx, "system:ingest", 1_000_000).Allowed {
		t.Fatal("a system consumer was held to the user budget")
	}
	m.SetTime(time.Date(2026, 10, 23, 0, 0, 1, 0, time.UTC))
	if !l.Admit(ctx, "user:alice", 1000).Allowed {
		t.Fatal("the budget didn't reset at midnight UTC")
	}
}

func TestReservationsAtOnceNeverOverspend(t *testing.T) {
	_, rdb := newRedis(t)
	var replicas []*Limiter
	for range 3 {
		l, _ := newLimiter(t, rdb, Settings{UserBudget: 10_000})
		replicas = append(replicas, l)
	}
	var reserved atomic.Int64
	var wg sync.WaitGroup
	for i := range 60 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if replicas[i%3].Admit(context.Background(), "user:alice", 300).Allowed {
				reserved.Add(300)
			}
		}()
	}
	wg.Wait()
	if got := reserved.Load(); got > 10_000 || got < 9_900 {
		t.Fatalf("reserved %d of a 10,000 budget", got)
	}
}

func TestWithoutRedisEachReplicaKeepsAThirdOfTheLimits(t *testing.T) {
	m, rdb := newRedis(t)
	l, metrics := newLimiter(t, rdb, Settings{GlobalRate: 0.001, GlobalBurst: 30, UserBudget: 900, Replicas: 3})
	ctx := context.Background()
	m.Close()

	admitted := 0
	for range 20 {
		if l.Admit(ctx, "user:alice", 0).Allowed {
			admitted++
		}
	}
	if admitted != 10 {
		t.Fatalf("%d admitted without Redis, want a third of 30", admitted)
	}
	if !l.Degraded() || !metrics.degraded.Load() {
		t.Fatal("limiter_degraded not set")
	}
	// The budget is divided too: 300 of 900
	l2, _ := newLimiter(t, rdb, Settings{UserBudget: 900, Replicas: 3})
	if !l2.Admit(ctx, "user:bob", 300).Allowed || l2.Admit(ctx, "user:bob", 1).Allowed {
		t.Fatal("the local budget isn't a third of the shared one")
	}

	// Back: the probe notices within a couple of seconds
	if err := m.Restart(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for l.Degraded() && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if l.Degraded() || metrics.degraded.Load() {
		t.Fatal("never went back to Redis")
	}
}

func TestLocalLimitsAloneWhenNoRedisIsConfigured(t *testing.T) {
	l, _ := newLimiter(t, nil, Settings{ConsumerRate: 1, ConsumerBurst: 3})
	n := 0
	for range 10 {
		if l.Admit(context.Background(), "user:alice", 0).Allowed {
			n++
		}
	}
	if n != 3 || l.Degraded() {
		t.Fatalf("admitted %d (degraded %v): a lone instance keeps the full limits", n, l.Degraded())
	}
}

// One test against a real Redis, with real time: set GATEWAY_TEST_REDIS_URL
// (CI runs a Redis service for it).
func TestTheGlobalLimitHoldsOnARealRedis(t *testing.T) {
	url := os.Getenv("GATEWAY_TEST_REDIS_URL")
	if url == "" {
		t.Skip("set GATEWAY_TEST_REDIS_URL to run against a real Redis")
	}
	opts, err := redis.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}
	rdb := redis.NewClient(opts)
	t.Cleanup(func() { _ = rdb.Close() })
	ctx := context.Background()
	if err := rdb.Del(ctx, globalKey).Err(); err != nil {
		t.Fatal(err)
	}
	rate, burst := 20.0, 40.0
	s := Settings{GlobalRate: rate, GlobalBurst: burst, Replicas: 3}
	var replicas []*Limiter
	for range 3 {
		l, _ := newLimiter(t, rdb, s)
		replicas = append(replicas, l)
	}
	start := time.Now()
	var admitted atomic.Int64
	var wg sync.WaitGroup
	for client := range 24 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 50 {
				if replicas[(client+i)%3].Admit(ctx, fmt.Sprintf("user:r%d", client), 0).Allowed {
					admitted.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(start).Seconds()
	bound := burst + rate*elapsed
	if got := float64(admitted.Load()); got > bound+1 {
		t.Fatalf("admitted %v in %.2fs; the global limit allows at most %.1f", got, elapsed, bound)
	}
	t.Logf("admitted %d of 1200 in %.2fs (limit %.1f)", admitted.Load(), elapsed, bound)
}

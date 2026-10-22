// Package limits decides whether a call may go to a provider: rate limits (a
// token bucket per consumer and one for everyone) and daily budgets per
// consumer, shared by every replica through Redis.
//
// Each decision is one Lua script, so checking and taking happen atomically
// however many replicas ask at once, and every script reads the time from
// Redis (TIME), never from a replica's clock: replicas that disagree about the
// time can't disagree about a bucket.
//
// Budgets are in weighted tokens (see package pricing) per UTC day. A call
// reserves an estimate before it goes out; Settle corrects the reservation to
// what the provider reported, or refunds it if the call failed.
//
// If Redis can't be reached, each replica falls back to limits of its own at a
// third of the configured ones (three replicas share the load), sets the
// limiter_degraded metric and logs once; it goes back to Redis when Redis
// answers again.
package limits

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

// Settings are the limits. A rate or budget of 0 turns that limit off.
type Settings struct {
	ConsumerRate  float64 // calls per second, per "user:" consumer
	ConsumerBurst float64
	GlobalRate    float64 // calls per second, everyone together
	GlobalBurst   float64
	UserBudget    int64 // weighted tokens per UTC day, per "user:" consumer
	SystemBudget  int64 // per "system:" consumer (ingest, eval)
	// Replicas is how many share the limits: the fallback divides by it.
	Replicas int
}

// Reason says which limit refused a call.
type Reason string

const (
	ConsumerRate Reason = "consumer_rate"
	GlobalRate   Reason = "global_rate"
	Budget       Reason = "budget"
)

// Decision is the answer for one call.
type Decision struct {
	Allowed    bool
	Reason     Reason        // when refused
	RetryAfter time.Duration // when refused
	// Reservation is what was reserved from the budget, to settle afterwards
	Reservation Reservation
}

// Reservation is budget taken for a call in flight.
type Reservation struct {
	consumer string
	day      int64
	amount   int64
	local    bool // taken from the fallback, not Redis
}

// Metrics the limiter reports.
type Metrics interface {
	SetDegraded(bool)
	Rejected(Reason)
}

// Limiter makes decisions.
type Limiter struct {
	rdb      redis.UniversalClient // nil: no Redis configured, local limits only
	settings Settings
	local    *local
	metrics  Metrics
	log      *slog.Logger

	degraded atomic.Bool
	stop     chan struct{}
}

// New returns a limiter on Redis, or on local limits alone if rdb is nil.
func New(rdb redis.UniversalClient, s Settings, m Metrics, log *slog.Logger) *Limiter {
	if s.Replicas < 1 {
		s.Replicas = 1
	}
	l := &Limiter{rdb: rdb, settings: s, metrics: m, log: log, stop: make(chan struct{})}
	if rdb == nil {
		l.local = newLocal(s) // a single instance: its own limits are the limits
	} else {
		l.local = newLocal(s.divided(s.Replicas))
	}
	return l
}

// Close stops the background Redis probe.
func (l *Limiter) Close() {
	select {
	case <-l.stop:
	default:
		close(l.stop)
	}
}

// Degraded is true while the limiter runs on local limits for lack of Redis.
func (l *Limiter) Degraded() bool { return l.degraded.Load() }

func (s Settings) divided(n int) Settings {
	d := float64(n)
	return Settings{
		ConsumerRate: s.ConsumerRate / d, ConsumerBurst: math.Max(1, math.Floor(s.ConsumerBurst/d)),
		GlobalRate: s.GlobalRate / d, GlobalBurst: math.Max(1, math.Floor(s.GlobalBurst/d)),
		UserBudget: ceilDiv(s.UserBudget, int64(n)), SystemBudget: ceilDiv(s.SystemBudget, int64(n)),
		Replicas: 1,
	}
}

func ceilDiv(a, b int64) int64 {
	if a <= 0 {
		return a
	}
	return (a + b - 1) / b
}

func isSystem(consumer string) bool { return strings.HasPrefix(consumer, "system:") }

func (s Settings) budgetFor(consumer string) int64 {
	if isSystem(consumer) {
		return s.SystemBudget
	}
	return s.UserBudget
}

// consumerRate is the per-consumer rate for this consumer. System consumers
// (ingest, the benchmarks) are jobs, not people: only the global bucket
// paces them, since a benchmark makes more calls a minute than any user may.
func (s Settings) consumerRate(consumer string) float64 {
	if isSystem(consumer) {
		return 0
	}
	return s.ConsumerRate
}

// FailFast sets a Redis client to give up quickly: a limiter waiting on Redis
// delays every call, and the fallback limits are only a timeout away.
func FailFast(opts *redis.Options) *redis.Options {
	opts.DialTimeout = 250 * time.Millisecond
	opts.ReadTimeout = 250 * time.Millisecond
	opts.WriteTimeout = 250 * time.Millisecond
	opts.MaxRetries = -1
	return opts
}

// The keys for a consumer. Consumer ids are only ever keys, never metric labels.
func bucketKey(consumer string) string { return "gw:rate:{" + consumer + "}" }
func budgetKey(consumer string) string { return "gw:budget:{" + consumer + "}" }

const globalKey = "gw:rate:global"

// admitScript checks the consumer's bucket, the global bucket and the
// consumer's budget, and takes from all three only if all three allow it.
// Returns {allowed, reason (1 consumer rate, 2 global rate, 3 budget),
// retry after in ms, the UTC day number}.
var admitScript = redis.NewScript(`
local t = redis.call('TIME')
local secs = tonumber(t[1])
local now = secs + tonumber(t[2]) / 1000000
local day = math.floor(secs / 86400)

local function level(key, rate, burst)
  if rate <= 0 then return nil end
  local v = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens, ts = tonumber(v[1]), tonumber(v[2])
  if tokens == nil or ts == nil then return burst end
  return math.min(burst, tokens + math.max(0, now - ts) * rate)
end

local crate, cburst = tonumber(ARGV[1]), tonumber(ARGV[2])
local grate, gburst = tonumber(ARGV[3]), tonumber(ARGV[4])
local limit, amount = tonumber(ARGV[5]), tonumber(ARGV[6])

local ctokens = level(KEYS[1], crate, cburst)
if ctokens ~= nil and ctokens < 1 then
  return {0, 1, math.ceil((1 - ctokens) / crate * 1000), day}
end
local gtokens = level(KEYS[2], grate, gburst)
if gtokens ~= nil and gtokens < 1 then
  return {0, 2, math.ceil((1 - gtokens) / grate * 1000), day}
end

if limit > 0 then
  local v = redis.call('HMGET', KEYS[3], 'day', 'used')
  local used = 0
  if tonumber(v[1]) == day then used = tonumber(v[2]) or 0 end
  if used + amount > limit then
    return {0, 3, ((day + 1) * 86400 - secs) * 1000, day}
  end
  redis.call('HSET', KEYS[3], 'day', day, 'used', used + amount)
  redis.call('EXPIRE', KEYS[3], 2 * 86400)
end

local function take(key, tokens, rate, burst)
  redis.call('HSET', key, 'tokens', string.format('%.6f', tokens - 1), 'ts', string.format('%.6f', now))
  redis.call('PEXPIRE', key, math.ceil(burst / rate * 1000) + 1000)
end
if ctokens ~= nil then take(KEYS[1], ctokens, crate, cburst) end
if gtokens ~= nil then take(KEYS[2], gtokens, grate, gburst) end
return {1, 0, 0, day}
`)

// settleScript applies (actual - reserved) to the day the reservation was
// made on. A reservation from a day that has since ended is left alone.
var settleScript = redis.NewScript(`
local v = redis.call('HMGET', KEYS[1], 'day', 'used')
if tonumber(v[1]) ~= tonumber(ARGV[1]) then return -1 end
local used = math.max(0, (tonumber(v[2]) or 0) + tonumber(ARGV[2]))
redis.call('HSET', KEYS[1], 'used', used)
return used
`)

// Admit decides on one call for `consumer` that is estimated to cost `estimate`
// weighted tokens, reserving them if it goes ahead.
func (l *Limiter) Admit(ctx context.Context, consumer string, estimate int64) Decision {
	limit := l.settings.budgetFor(consumer)
	if l.rdb == nil || l.degraded.Load() {
		d := l.local.admit(consumer, estimate, time.Now())
		l.report(d)
		return d
	}
	s := l.settings
	res, err := admitScript.Run(ctx, l.rdb,
		[]string{bucketKey(consumer), globalKey, budgetKey(consumer)},
		s.consumerRate(consumer), s.ConsumerBurst, s.GlobalRate, s.GlobalBurst, limit, estimate,
	).Int64Slice()
	if err != nil || len(res) != 4 {
		if err == nil {
			err = errors.New("unexpected reply from the admit script")
		}
		l.lostRedis(err)
		d := l.local.admit(consumer, estimate, time.Now())
		l.report(d)
		return d
	}
	d := Decision{Allowed: res[0] == 1}
	if d.Allowed {
		if limit > 0 {
			d.Reservation = Reservation{consumer: consumer, day: res[3], amount: estimate}
		}
	} else {
		d.Reason = [...]Reason{"", ConsumerRate, GlobalRate, Budget}[res[1]]
		d.RetryAfter = time.Duration(res[2]) * time.Millisecond
	}
	l.report(d)
	return d
}

// Settle corrects a reservation to what the call actually cost: 0 refunds it
// (the call failed), a negative number keeps the estimate (cost unknown).
func (l *Limiter) Settle(ctx context.Context, r Reservation, actual int64) {
	if r.amount == 0 && r.consumer == "" {
		return
	}
	if actual < 0 {
		return
	}
	delta := actual - r.amount
	if delta == 0 {
		return
	}
	if r.local || l.rdb == nil {
		l.local.settle(r, delta)
		return
	}
	// Best effort even while degraded: the reservation lives in Redis
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
	defer cancel()
	if err := settleScript.Run(ctx, l.rdb, []string{budgetKey(r.consumer)}, r.day, delta).Err(); err != nil {
		l.lostRedis(err)
	}
}

func (l *Limiter) report(d Decision) {
	if !d.Allowed && l.metrics != nil {
		l.metrics.Rejected(d.Reason)
	}
}

// lostRedis switches to local limits, once, and starts probing for Redis.
func (l *Limiter) lostRedis(err error) {
	if l.degraded.CompareAndSwap(false, true) {
		l.log.Error("limiter: Redis unreachable, using per-replica limits at 1/"+
			strconv.Itoa(l.settings.Replicas)+" until it is back", "error", err.Error())
		if l.metrics != nil {
			l.metrics.SetDegraded(true)
		}
		go l.probe()
	}
}

// probe pings Redis every second while degraded, and switches back when it answers.
func (l *Limiter) probe() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-l.stop:
			return
		case <-ticker.C:
		}
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
		err := l.rdb.Ping(ctx).Err()
		cancel()
		if err == nil {
			l.degraded.Store(false)
			if l.metrics != nil {
				l.metrics.SetDegraded(false)
			}
			l.log.Info("limiter: Redis is back, shared limits restored")
			return
		}
	}
}

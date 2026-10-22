package limits

import (
	"math"
	"sync"
	"time"
)

// local is the same limits kept in this process: what a replica falls back to
// without Redis (at a third of the limits), or uses alone when none is set up.
type local struct {
	mu       sync.Mutex
	settings Settings
	buckets  map[string]*bucket
	global   bucket
	budgets  map[string]*spent
}

type bucket struct {
	tokens float64
	at     time.Time
}

type spent struct {
	day  int64
	used int64
}

func newLocal(s Settings) *local {
	return &local{settings: s, buckets: map[string]*bucket{}, budgets: map[string]*spent{}}
}

// level is the bucket's tokens now (nil bucket: full).
func level(b *bucket, rate, burst float64, now time.Time) float64 {
	if b == nil || b.at.IsZero() {
		return burst
	}
	return math.Min(burst, b.tokens+math.Max(0, now.Sub(b.at).Seconds())*rate)
}

func wait(tokens, rate float64) time.Duration {
	return time.Duration(math.Ceil((1-tokens)/rate*1000)) * time.Millisecond
}

func (l *local) admit(consumer string, estimate int64, now time.Time) Decision {
	l.mu.Lock()
	defer l.mu.Unlock()
	s := l.settings
	if len(l.buckets) > 10000 {
		l.prune(now)
	}

	c := l.buckets[consumer]
	rate := s.consumerRate(consumer)
	var ctokens, gtokens float64
	if rate > 0 {
		ctokens = level(c, rate, s.ConsumerBurst, now)
		if ctokens < 1 {
			return Decision{Reason: ConsumerRate, RetryAfter: wait(ctokens, rate)}
		}
	}
	if s.GlobalRate > 0 {
		gtokens = level(&l.global, s.GlobalRate, s.GlobalBurst, now)
		if gtokens < 1 {
			return Decision{Reason: GlobalRate, RetryAfter: wait(gtokens, s.GlobalRate)}
		}
	}

	var r Reservation
	if limit := s.budgetFor(consumer); limit > 0 {
		day := now.UTC().Unix() / 86400
		b := l.budgets[consumer]
		if b == nil || b.day != day {
			b = &spent{day: day}
			l.budgets[consumer] = b
		}
		if b.used+estimate > limit {
			return Decision{Reason: Budget, RetryAfter: time.Duration((day+1)*86400-now.UTC().Unix()) * time.Second}
		}
		b.used += estimate
		r = Reservation{consumer: consumer, day: day, amount: estimate, local: true}
	}
	if rate > 0 {
		if c == nil {
			c = &bucket{}
			l.buckets[consumer] = c
		}
		c.tokens, c.at = ctokens-1, now
	}
	if s.GlobalRate > 0 {
		l.global.tokens, l.global.at = gtokens-1, now
	}
	return Decision{Allowed: true, Reservation: r}
}

func (l *local) settle(r Reservation, delta int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if b := l.budgets[r.consumer]; b != nil && b.day == r.day {
		b.used = max(0, b.used+delta)
	}
}

// prune forgets buckets that have refilled, so the map can't grow without end.
func (l *local) prune(now time.Time) {
	for k, b := range l.buckets {
		if level(b, l.settings.ConsumerRate, l.settings.ConsumerBurst, now) >= l.settings.ConsumerBurst {
			delete(l.buckets, k)
		}
	}
	day := now.UTC().Unix() / 86400
	for k, b := range l.budgets {
		if b.day != day {
			delete(l.budgets, k)
		}
	}
}

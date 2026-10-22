// Package breaker is a circuit breaker for one provider, in one replica.
//
//	closed     calls go through; Failures failures in a row open it
//	open       calls skip this provider until Cooldown has passed
//	half-open  one call goes through as a probe: success closes it,
//	           failure opens it again for another Cooldown
//
// What counts as a failure is the caller's call (see server.provider.go):
// 5xx, connection errors, timeouts and the provider's own 429s; never another
// 4xx, which is the request's fault, not the provider's.
package breaker

import (
	"sync"
	"time"
)

// State is a breaker's state.
type State int

const (
	Closed State = iota
	HalfOpen
	Open
)

func (s State) String() string {
	return [...]string{"closed", "half-open", "open"}[s]
}

// Breaker is safe for concurrent use.
type Breaker struct {
	mu        sync.Mutex
	failures  int // in a row
	threshold int
	cooldown  time.Duration
	state     State
	openedAt  time.Time
	probing   bool // a half-open probe is in flight
	now       func() time.Time
	onChange  func(State)
}

// New returns a closed breaker. onChange (may be nil) hears every transition.
func New(threshold int, cooldown time.Duration, onChange func(State)) *Breaker {
	if threshold < 1 {
		threshold = 1
	}
	return &Breaker{threshold: threshold, cooldown: cooldown, now: time.Now, onChange: onChange}
}

// Allow says whether a call may go to this provider now. In half-open state
// only one caller at a time is allowed: the probe.
func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch b.state {
	case Closed:
		return true
	case Open:
		if b.now().Sub(b.openedAt) < b.cooldown {
			return false
		}
		b.set(HalfOpen)
		fallthrough
	default: // half-open
		if b.probing {
			return false
		}
		b.probing = true
		return true
	}
}

// Success records a call the provider answered.
func (b *Breaker) Success() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures = 0
	b.probing = false
	if b.state != Closed {
		b.set(Closed)
	}
}

// Failure records a call the provider failed.
func (b *Breaker) Failure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	wasProbe := b.state == HalfOpen
	b.probing = false
	if wasProbe || (b.state == Closed && b.failures >= b.threshold) {
		b.openedAt = b.now()
		b.set(Open)
	}
}

// Release gives back a half-open probe that ended without a verdict (the
// caller hung up before the provider answered).
func (b *Breaker) Release() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.probing = false
}

// State is the current state (an open breaker past its cooldown reads half-open).
func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == Open && b.now().Sub(b.openedAt) >= b.cooldown {
		return HalfOpen
	}
	return b.state
}

func (b *Breaker) set(s State) {
	b.state = s
	if b.onChange != nil {
		b.onChange(s)
	}
}

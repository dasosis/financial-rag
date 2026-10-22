package breaker

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func newTest(threshold int) (*Breaker, *clock, *[]State) {
	c := &clock{t: time.Unix(1_000_000, 0)}
	var changes []State
	b := New(threshold, 30*time.Second, func(s State) { changes = append(changes, s) })
	b.now = c.now
	return b, c, &changes
}

func TestItOpensAfterFailuresInARowAndProbesAfterTheCooldown(t *testing.T) {
	b, c, changes := newTest(3)
	b.Failure()
	b.Failure()
	b.Success() // a success in between resets the count
	b.Failure()
	b.Failure()
	if !b.Allow() || b.State() != Closed {
		t.Fatal("opened before three failures in a row")
	}
	b.Failure()
	if b.Allow() || b.State() != Open {
		t.Fatal("not open after three in a row")
	}
	c.t = c.t.Add(29 * time.Second)
	if b.Allow() {
		t.Fatal("allowed during the cooldown")
	}
	c.t = c.t.Add(time.Second)
	if !b.Allow() {
		t.Fatal("no probe after the cooldown")
	}
	if b.Allow() {
		t.Fatal("a second caller was let through while the probe is out")
	}
	b.Success()
	if b.State() != Closed || !b.Allow() {
		t.Fatal("a successful probe didn't close it")
	}
	want := []State{Open, HalfOpen, Closed}
	if len(*changes) != len(want) {
		t.Fatalf("changes %v", *changes)
	}
	for i := range want {
		if (*changes)[i] != want[i] {
			t.Fatalf("changes %v, want %v", *changes, want)
		}
	}
}

func TestAFailedProbeOpensItAgain(t *testing.T) {
	b, c, _ := newTest(1)
	b.Failure()
	c.t = c.t.Add(30 * time.Second)
	if !b.Allow() {
		t.Fatal("no probe")
	}
	b.Failure()
	if b.State() != Open || b.Allow() {
		t.Fatal("a failed probe didn't reopen it")
	}
	c.t = c.t.Add(30 * time.Second)
	if !b.Allow() {
		t.Fatal("no second probe after another cooldown")
	}
}

func TestAReleasedProbeLetsTheNextOneThrough(t *testing.T) {
	b, c, _ := newTest(1)
	b.Failure()
	c.t = c.t.Add(30 * time.Second)
	b.Allow()
	b.Release()
	if !b.Allow() {
		t.Fatal("the probe slot wasn't given back")
	}
}

func TestOnlyOneProbeAtATime(t *testing.T) {
	b, c, _ := newTest(1)
	b.Failure()
	c.t = c.t.Add(time.Minute)
	var allowed atomic.Int32
	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if b.Allow() {
				allowed.Add(1)
			}
		}()
	}
	wg.Wait()
	if allowed.Load() != 1 {
		t.Fatalf("%d probes let through at once", allowed.Load())
	}
}

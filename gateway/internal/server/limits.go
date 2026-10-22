package server

import (
	"math"
	"net/http"
	"strconv"

	"github.com/dasosis/financial-rag/gateway/internal/limits"
	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

// admit asks the limiter whether the call may go ahead, reserving its
// estimated cost from the consumer's budget. A refusal is answered here: 429
// with Retry-After and x-should-retry: false, which the app turns into its own
// 429 for the user (and refunds the question).
func (s *Server) admit(c *call, model string, estimate usage.Usage) (limits.Reservation, bool) {
	if s.limiter == nil {
		return limits.Reservation{}, true
	}
	d := s.limiter.Admit(c.r.Context(), c.consumer, s.prices.Units(model, estimate))
	if d.Allowed {
		return d.Reservation, true
	}
	retry := max(1, int(math.Ceil(d.RetryAfter.Seconds())))
	c.w.Header().Set("Retry-After", strconv.Itoa(retry))
	if d.Reason == limits.Budget {
		s.fail(c, http.StatusTooManyRequests, "rate_limit_error", CodeBudgetExhausted,
			"This consumer's daily budget is spent; it resets at midnight UTC.")
	} else {
		s.fail(c, http.StatusTooManyRequests, "rate_limit_error", CodeRateLimited,
			"Too many calls; try again after Retry-After seconds.")
	}
	return limits.Reservation{}, false
}

// settle corrects the reservation to what the call cost in budget units: 0
// refunds it, a negative number (cost unknown) keeps the estimate.
func (s *Server) settle(c *call, r limits.Reservation, units int64) {
	if s.limiter != nil {
		s.limiter.Settle(c.r.Context(), r, units)
	}
}

// account counts a call's tokens and cost, and returns its cost in budget
// units, or -1 if the provider reported no usage (a stream cut short).
func (s *Server) account(model string, u usage.Usage) int64 {
	if !u.Known() {
		return -1
	}
	if u.PromptTokens > 0 {
		s.metrics.Tokens.WithLabelValues(model, "prompt").Add(float64(u.PromptTokens))
	}
	if u.CompletionTokens > 0 {
		s.metrics.Tokens.WithLabelValues(model, "completion").Add(float64(u.CompletionTokens))
	}
	if n := u.Cached(); n > 0 {
		s.metrics.Tokens.WithLabelValues(model, "cached").Add(float64(n))
	}
	if _, known := s.prices.Lookup(model); !known {
		s.warnUnpriced(model)
	}
	s.metrics.Cost.WithLabelValues(model).Add(s.prices.Cost(model, u))
	return s.prices.Units(model, u)
}

// warnUnpriced logs, once per model, that it is charged at the fallback price.
func (s *Server) warnUnpriced(model string) {
	if _, seen := s.unpriced.LoadOrStore(model, true); !seen {
		s.log.Warn("no price for this model: charged as gpt-4o until GATEWAY_PRICES has it",
			"model", model)
	}
}

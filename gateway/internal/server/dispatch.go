package server

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"
)

// outbound is a call as one provider must receive it: the body (with that
// provider's model name) and whether it streams.
type outbound struct {
	body   []byte
	stream bool
}

// A plan says how to send the call to a provider, or false if it can't take it
// (it doesn't serve the model, or can't follow the request's json_schema).
type plan func(p *Provider) (outbound, bool)

// dispatch sends the call to the first provider that takes it: the primary,
// then the backup, skipping any whose breaker is open. A provider that fails
// before it starts answering (connection error, no first byte in time, 5xx, or
// its own 429) counts against its breaker and the next one is tried; nothing
// has gone back to the caller yet, so switching is safe. Once a provider has
// started answering, the call is theirs: the gateway never switches after the
// first byte. Other 4xx are the request's fault and are passed on as they are.
//
// When no provider takes the call it is answered here (503
// providers_unavailable, or 400 for an unknown X-Force-Provider) and dispatch
// returns ok false.
func (s *Server) dispatch(c *call, path string, how plan) (*upstreamResponse, *Provider, outbound, bool) {
	providers := s.providers
	forced := c.r.Header.Get("X-Force-Provider")
	if forced != "" {
		// Only callers holding the internal key reach this point (endpoint());
		// the benchmarks use it to score the backup on its own.
		providers = nil
		for _, p := range s.providers {
			if p.Name == forced {
				providers = append(providers, p)
			}
		}
		if len(providers) == 0 {
			s.fail(c, http.StatusBadRequest, "invalid_request_error", "unknown_provider",
				"X-Force-Provider names no configured provider.")
			return nil, nil, outbound{}, false
		}
	}

	var candidates []*Provider
	var plans []outbound
	for _, p := range providers {
		if out, ok := how(p); ok {
			candidates = append(candidates, p)
			plans = append(plans, out)
		}
	}

	failedOver := false
	for i, p := range candidates {
		if forced == "" && !p.Breaker.Allow() {
			continue
		}
		last := i == len(candidates)-1
		for attempt := 0; ; attempt++ {
			resp, retry, err := s.try(c, p, path, plans[i])
			if err == nil {
				if failedOver {
					s.metrics.Failovers.WithLabelValues(c.endpoint).Inc()
				}
				c.w.Header().Set("X-Gateway-Provider", p.Name)
				return resp, p, plans[i], true
			}
			if c.r.Context().Err() != nil {
				c.w.code = 499 // the caller hung up; there is no one to answer
				return nil, nil, outbound{}, false
			}
			// With nowhere else to go, a blip (a dropped connection, a 502) gets
			// one more try, as the app's own client would have retried OpenAI
			if retry && attempt == 0 && last && !failedOver && (forced != "" || p.Breaker.Allow()) {
				time.Sleep(200 * time.Millisecond)
				continue
			}
			break
		}
		failedOver = true
	}
	s.fail(c, http.StatusServiceUnavailable, "api_error", CodeProvidersUnavailable,
		"No model provider could take the call.")
	return nil, nil, outbound{}, false
}

// try makes one call to one provider and reports it to the provider's breaker.
// err is nil when the provider answered (2xx, or a 4xx other than 429); retry
// says whether the failure was a blip worth trying again.
func (s *Server) try(c *call, p *Provider, path string, out outbound) (*upstreamResponse, bool, error) {
	firstByte := s.cfg.ChatTimeout
	switch {
	case c.endpoint == "embeddings":
		firstByte = s.cfg.EmbedTimeout
	case out.stream:
		firstByte = s.cfg.StreamTTFBTimeout
	}
	sent := time.Now()
	resp, err := p.send(c.r.Context(), path, out.body, firstByte, out.stream)
	if err != nil {
		s.metrics.Upstream.WithLabelValues(p.Name, c.endpoint, "error").Inc()
		if c.r.Context().Err() != nil {
			p.Breaker.Release() // no verdict on the provider
			return nil, false, err
		}
		p.Breaker.Failure()
		timeout := errors.Is(err, errFirstByteTimeout) || errors.Is(err, context.DeadlineExceeded)
		s.log.Warn("provider failed", "request_id", c.requestID, "provider", p.Name,
			"endpoint", c.endpoint, "timeout", timeout, "error", err.Error())
		return nil, !timeout, err // a timeout already waited long enough
	}
	s.metrics.UpLatency.WithLabelValues(p.Name, c.endpoint).Observe(time.Since(sent).Seconds())
	s.metrics.Upstream.WithLabelValues(p.Name, c.endpoint, strconv.Itoa(resp.StatusCode)).Inc()
	if resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests {
		resp.Close()
		p.Breaker.Failure()
		s.log.Warn("provider failed", "request_id", c.requestID, "provider", p.Name,
			"endpoint", c.endpoint, "status", resp.StatusCode)
		return nil, resp.StatusCode != http.StatusTooManyRequests,
			errors.New("provider status " + strconv.Itoa(resp.StatusCode))
	}
	p.Breaker.Success()
	return resp, false, nil
}

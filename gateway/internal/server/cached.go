package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/cache"
	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

// lookup is what the cache said about a call.
type lookup struct {
	key   string
	hit   []byte // the stored answer, on a hit
	store bool   // a miss worth storing once answered
}

// cacheLookup looks the call up, unless it can't be cached (not temperature 0)
// or the caller said Cache-Control: no-cache, as the benchmarks do. no-store
// keeps an answer out of the cache as well.
func (s *Server) cacheLookup(c *call, req *chatRequest) lookup {
	if s.cache == nil {
		return lookup{}
	}
	key, cacheable := cache.Key(req.fields)
	if !cacheable {
		return lookup{}
	}
	directives := cacheControl(c.r.Header)
	l := lookup{key: key, store: !directives["no-store"]}
	if directives["no-cache"] || directives["no-store"] {
		c.w.Header().Set("X-Gateway-Cache", "bypass")
		s.metrics.Cache.WithLabelValues("bypass").Inc()
		return l
	}
	body, hit, err := s.cache.Get(c.r.Context(), key)
	if err != nil {
		s.cacheTrouble(err)
	}
	if hit {
		l.hit = body
		s.metrics.Cache.WithLabelValues("hit").Inc()
		return l
	}
	c.w.Header().Set("X-Gateway-Cache", "miss")
	s.metrics.Cache.WithLabelValues("miss").Inc()
	return l
}

func cacheControl(h http.Header) map[string]bool {
	out := map[string]bool{}
	for _, value := range h.Values("Cache-Control") {
		for _, d := range strings.Split(value, ",") {
			out[strings.ToLower(strings.TrimSpace(d))] = true
		}
	}
	return out
}

// serveCached answers from a stored answer: as JSON, or replayed as SSE for a
// streamed call. Nothing goes to the provider, and nothing is charged to the
// consumer's budget: the metric counts what it would have cost.
func (s *Server) serveCached(c *call, req *chatRequest, body []byte) {
	h := c.w.Header()
	h.Set("X-Gateway-Cache", "hit")
	var stored struct {
		Usage *usage.Usage `json:"usage"`
	}
	if json.Unmarshal(body, &stored) == nil && stored.Usage != nil {
		s.metrics.CacheSaved.WithLabelValues(req.model).Add(s.prices.Cost(req.model, *stored.Usage))
	}
	if req.stream {
		events, err := cache.Replay(body, req.clientUsage)
		if err != nil { // can't happen for what Put stores; answer as a provider error would
			s.fail(c, http.StatusBadGateway, "api_error", "upstream_unavailable", "Couldn't replay the cached answer.")
			return
		}
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no")
		c.w.WriteHeader(http.StatusOK)
		_, _ = c.w.Write(events)
		c.w.Flush()
	} else {
		h.Set("Content-Type", "application/json")
		c.w.WriteHeader(http.StatusOK)
		_, _ = c.w.Write(body)
	}
	s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint, "stream", req.stream,
		"model", req.model, "cache", "hit", "ms", time.Since(c.start).Milliseconds())
}

// cacheStore keeps a finished answer (cache.Put refuses anything else).
func (s *Server) cacheStore(c *call, key string, body []byte) {
	if err := s.cache.Put(c.r.Context(), key, body); err != nil {
		s.cacheTrouble(err)
	}
}

// cacheTrouble logs a cache Redis failure, at most once a minute: the cache is
// an optimisation, and a call without it is merely slower and dearer.
func (s *Server) cacheTrouble(err error) {
	now := time.Now().Unix()
	last := s.cacheWarned.Load()
	if now-last >= 60 && s.cacheWarned.CompareAndSwap(last, now) {
		s.log.Warn("response cache unavailable, answering without it", "error", err.Error())
	}
}

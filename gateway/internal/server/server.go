// Package server is the gateway's HTTP side: the OpenAI-compatible endpoints
// the app calls instead of OpenAI, and health checks.
//
//	POST /v1/chat/completions   plain and streamed (SSE)
//	POST /v1/embeddings
//	GET  /healthz               the process is up
//	GET  /readyz                it is taking requests (false while shutting down)
//
// Every /v1 call needs `Authorization: Bearer <LLM_GATEWAY_KEY>` and an
// `X-Consumer-Id` naming who it is for ("user:<id>", "system:ingest"), or it
// gets 401. Errors are OpenAI-shaped JSON, {"error": {"message", "type", "code"}},
// so the OpenAI client raises its usual exceptions.
package server

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/breaker"
	"github.com/dasosis/financial-rag/gateway/internal/cache"
	"github.com/dasosis/financial-rag/gateway/internal/config"
	"github.com/dasosis/financial-rag/gateway/internal/limits"
	"github.com/dasosis/financial-rag/gateway/internal/metrics"
	"github.com/dasosis/financial-rag/gateway/internal/pricing"
	"github.com/dasosis/financial-rag/gateway/internal/redact"
)

// Error codes the app maps to its own exceptions (financial_rag/config.py).
const (
	CodeBudgetExhausted      = "budget_exhausted"
	CodeRateLimited          = "rate_limited"
	CodeProvidersUnavailable = "providers_unavailable"
)

// A consumer id: a kind and a name, "user:3f2c…" or "system:eval".
var consumerRE = regexp.MustCompile(`^(user|system):[A-Za-z0-9._:-]{1,128}$`)

// Server answers the gateway's endpoints.
type Server struct {
	cfg     config.Config
	metrics *metrics.Metrics
	log     *slog.Logger
	// The providers in the order calls try them: OpenAI, then the backup
	providers []*Provider
	limiter   *limits.Limiter
	cache     *cache.Cache
	redactor  *redact.Redactor // nil: redaction off
	prices    *pricing.Table
	ready     atomic.Bool
	unpriced  sync.Map // models already warned about
	// when the cache's trouble was last logged (unix seconds)
	cacheWarned atomic.Int64
}

// Deps are what a server works with besides its configuration.
type Deps struct {
	Metrics *metrics.Metrics
	Log     *slog.Logger
	Limiter *limits.Limiter // nil: no limits or budgets
	Cache   *cache.Cache    // nil: no response cache
	// The personal data to redact from prompts (package redact); nil: none
	Redactor *redact.Redactor
	Prices   *pricing.Table // nil: the built-in price table
}

// New builds a server that sends calls to OpenAI, and to OpenRouter when
// OpenAI fails if it has a key for it.
func New(cfg config.Config, d Deps) *Server {
	if d.Prices == nil {
		d.Prices, _ = pricing.New("")
	}
	s := &Server{
		cfg: cfg, metrics: d.Metrics, log: d.Log, limiter: d.Limiter, cache: d.Cache, prices: d.Prices,
		redactor: d.Redactor,
	}
	s.providers = []*Provider{NewProvider("openai", cfg.OpenAIBaseURL, cfg.OpenAIKey, nil, s.breaker("openai"))}
	if cfg.OpenRouterKey != "" {
		backup := NewProvider("openrouter", cfg.OpenRouterBaseURL, cfg.OpenRouterKey, cfg.BackupModels,
			s.breaker("openrouter"))
		backup.JSONSchema = cfg.BackupJSONSchema
		s.providers = append(s.providers, backup)
	}
	s.ready.Store(true)
	return s
}

// breaker makes a provider's breaker, reporting its state as a metric.
func (s *Server) breaker(provider string) *breaker.Breaker {
	gauge := s.metrics.Breaker.WithLabelValues(provider)
	gauge.Set(float64(breaker.Closed))
	return breaker.New(s.cfg.BreakerFailures, s.cfg.BreakerCooldown, func(state breaker.State) {
		gauge.Set(float64(state))
		s.log.Warn("circuit breaker", "provider", provider, "state", state.String())
	})
}

// SetReady marks the server ready or not; /readyz reports it, so the load
// balancer stops sending work to a replica that is shutting down.
func (s *Server) SetReady(ready bool) { s.ready.Store(ready) }

// Handler routes the API.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !s.ready.Load() {
			http.Error(w, "shutting down", http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte("ready\n"))
	})
	mux.Handle("POST /v1/chat/completions", s.endpoint("chat", s.chat))
	mux.Handle("POST /v1/embeddings", s.endpoint("embeddings", s.embeddings))
	return mux
}

// call is one authenticated request on its way through the gateway.
type call struct {
	endpoint  string
	consumer  string
	requestID string
	start     time.Time
	w         *recorder
	r         *http.Request
}

// endpoint wraps a handler with authentication, a request id and metrics.
func (s *Server) endpoint(name string, handle func(*call)) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := &call{endpoint: name, requestID: newRequestID(), start: time.Now(), r: r}
		c.w = &recorder{ResponseWriter: w, start: c.start}
		c.w.Header().Set("X-Request-Id", c.requestID)
		s.metrics.InFlight.Inc()
		defer func() {
			s.metrics.InFlight.Dec()
			code := strconv.Itoa(c.w.status())
			s.metrics.Requests.WithLabelValues(name, code).Inc()
			s.metrics.Latency.WithLabelValues(name).Observe(time.Since(c.start).Seconds())
			if !c.w.firstByte.IsZero() {
				s.metrics.TTFB.WithLabelValues(name).Observe(c.w.firstByte.Sub(c.start).Seconds())
			}
		}()

		if !s.authorized(r) {
			s.fail(c, http.StatusUnauthorized, "invalid_request_error", "unauthorized",
				"A valid gateway key is required.")
			return
		}
		c.consumer = r.Header.Get("X-Consumer-Id")
		if !consumerRE.MatchString(c.consumer) {
			s.fail(c, http.StatusUnauthorized, "invalid_request_error", "missing_consumer",
				"X-Consumer-Id must name the consumer, e.g. user:<id> or system:<job>.")
			return
		}
		handle(c)
	})
}

// authorized is true when the request carries the gateway key. Compared in
// constant time, so the key can't be guessed from response timings.
func (s *Server) authorized(r *http.Request) bool {
	scheme, token, ok := strings.Cut(r.Header.Get("Authorization"), " ")
	if !ok || !strings.EqualFold(scheme, "bearer") {
		return false
	}
	token = strings.TrimSpace(token)
	return subtle.ConstantTimeCompare([]byte(token), []byte(s.cfg.Key)) == 1
}

// fail answers with an OpenAI-style error.
func (s *Server) fail(c *call, status int, typ, code, message string) {
	h := c.w.Header()
	h.Set("Content-Type", "application/json")
	if status == http.StatusTooManyRequests || code == CodeProvidersUnavailable {
		// The app's client must not retry these: retrying a spent budget or an
		// outage only adds load. It tells the user instead.
		h.Set("x-should-retry", "false")
	}
	c.w.WriteHeader(status)
	_ = json.NewEncoder(c.w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": typ, "code": code, "param": nil},
	})
	s.log.Info("refused", "request_id", c.requestID, "endpoint", c.endpoint,
		"status", status, "code", code)
}

func newRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "gw-" + hex.EncodeToString(b[:])
}

// recorder notes the status sent and when the first byte went out.
type recorder struct {
	http.ResponseWriter
	start     time.Time
	code      int
	firstByte time.Time
}

func (r *recorder) WriteHeader(code int) {
	if r.code == 0 {
		r.code = code
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *recorder) Write(p []byte) (int, error) {
	if r.code == 0 {
		r.code = http.StatusOK
	}
	if r.firstByte.IsZero() && len(p) > 0 {
		r.firstByte = time.Now()
	}
	return r.ResponseWriter.Write(p)
}

// Flush sends buffered bytes now: streamed tokens must not wait in a buffer.
func (r *recorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.ResponseController reach the real writer.
func (r *recorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (r *recorder) status() int {
	if r.code == 0 {
		return http.StatusOK
	}
	return r.code
}

func (r *recorder) started() bool { return r.code != 0 }

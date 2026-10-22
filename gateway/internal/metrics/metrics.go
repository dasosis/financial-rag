// Package metrics holds the gateway's Prometheus metrics. Labels are
// endpoints, status codes, models and providers: never a consumer or user id,
// which would make one series per user and identify who asked what.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/dasosis/financial-rag/gateway/internal/limits"
)

// Latency buckets in seconds, up to two minutes: figure extraction calls run long.
var latencyBuckets = []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 90, 120}

// Metrics is one registry and the gateway's collectors in it.
type Metrics struct {
	Registry *prometheus.Registry

	Requests  *prometheus.CounterVec   // endpoint, code
	Latency   *prometheus.HistogramVec // endpoint
	TTFB      *prometheus.HistogramVec // endpoint: time to the first byte sent back
	InFlight  prometheus.Gauge
	Tokens    *prometheus.CounterVec // model, kind (prompt, completion, cached)
	Upstream  *prometheus.CounterVec // provider, endpoint, code: calls to providers
	UpLatency *prometheus.HistogramVec

	Cost            *prometheus.CounterVec // model: dollars at list price
	LimiterDegraded prometheus.Gauge
	Rejections      *prometheus.CounterVec // reason: consumer_rate, global_rate, budget

	Cache      *prometheus.CounterVec // result: hit, miss, bypass
	CacheSaved *prometheus.CounterVec // model: what hits would have cost
}

// New registers the gateway's metrics, plus Go runtime and process metrics.
func New() *Metrics {
	m := &Metrics{
		Registry: prometheus.NewRegistry(),
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "Requests answered, by endpoint and HTTP status code.",
		}, []string{"endpoint", "code"}),
		Latency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "Time to answer a request in full, by endpoint.",
			Buckets: latencyBuckets,
		}, []string{"endpoint"}),
		TTFB: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_time_to_first_byte_seconds",
			Help:    "Time until the first byte of a response went back to the caller.",
			Buckets: latencyBuckets,
		}, []string{"endpoint"}),
		InFlight: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gateway_in_flight_requests",
			Help: "Requests being answered now.",
		}),
		Tokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_tokens_total",
			Help: "Tokens providers reported using, by model and kind (prompt, completion, cached).",
		}, []string{"model", "kind"}),
		Upstream: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_upstream_requests_total",
			Help: "Calls to providers, by provider, endpoint and HTTP status (\"error\" when none came back).",
		}, []string{"provider", "endpoint", "code"}),
		UpLatency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_upstream_duration_seconds",
			Help:    "Time providers took to start answering (headers received), by provider and endpoint.",
			Buckets: latencyBuckets,
		}, []string{"provider", "endpoint"}),
		Cost: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cost_usd_total",
			Help: "What the calls cost at list price (the price table), by model.",
		}, []string{"model"}),
		LimiterDegraded: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gateway_limiter_degraded",
			Help: "1 while this replica can't reach the limits' Redis and keeps limits of its own.",
		}),
		Rejections: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_limit_rejections_total",
			Help: "Calls refused by a limit: consumer_rate, global_rate or budget.",
		}, []string{"reason"}),
		Cache: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cache_requests_total",
			Help: "Cacheable chat calls by result: hit, miss, or bypass (Cache-Control: no-cache).",
		}, []string{"result"}),
		CacheSaved: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cache_saved_usd_total",
			Help: "What the calls answered from the cache would have cost, at list price, by model.",
		}, []string{"model"}),
	}
	m.Registry.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
		m.Requests, m.Latency, m.TTFB, m.InFlight, m.Tokens, m.Upstream, m.UpLatency,
		m.Cost, m.LimiterDegraded, m.Rejections, m.Cache, m.CacheSaved,
	)
	for _, result := range []string{"hit", "miss", "bypass"} {
		m.Cache.WithLabelValues(result)
	}
	for _, reason := range []limits.Reason{limits.ConsumerRate, limits.GlobalRate, limits.Budget} {
		m.Rejections.WithLabelValues(string(reason)) // show 0 before the first
	}
	return m
}

// SetDegraded and Rejected report for the limiter (limits.Metrics).
func (m *Metrics) SetDegraded(on bool) {
	if on {
		m.LimiterDegraded.Set(1)
	} else {
		m.LimiterDegraded.Set(0)
	}
}

func (m *Metrics) Rejected(r limits.Reason) { m.Rejections.WithLabelValues(string(r)).Inc() }

// Handler serves /metrics.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.Registry, promhttp.HandlerOpts{})
}

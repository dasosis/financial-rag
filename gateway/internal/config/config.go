// Package config reads the gateway's settings from the environment.
package config

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/limits"
	"github.com/dasosis/financial-rag/gateway/internal/redact"
)

// Config is everything the gateway is told at start-up.
type Config struct {
	Addr        string // the OpenAI-compatible API
	MetricsAddr string // /metrics, internal only

	// The bearer key every caller must present. The app's LLM_GATEWAY_KEY.
	Key string

	OpenAIKey     string
	OpenAIBaseURL string

	// The backup provider, tried when OpenAI fails (package server, dispatch).
	// Off without OPENROUTER_API_KEY. BackupModels maps a model the app asks for
	// to the backup's name for it; a model it doesn't list never fails over.
	OpenRouterKey     string
	OpenRouterBaseURL string
	BackupModels      map[string]string
	BackupJSONSchema  bool // the backup can follow response_format json_schema

	// A provider's breaker opens after BreakerFailures failures in a row, and
	// lets one probe through after BreakerCooldown.
	BreakerFailures int
	BreakerCooldown time.Duration

	// How long a call may wait for the provider's first byte. Non-streamed
	// chat calls only send theirs when the whole answer is written, and figure
	// extraction can take minutes, so they get the most.
	ChatTimeout       time.Duration
	StreamTTFBTimeout time.Duration
	EmbedTimeout      time.Duration

	MaxBodyBytes int64

	// Limits and budgets (package limits). Unset LimitsRedisURL: each replica
	// keeps its own, which is only right for a single one.
	LimitsRedisURL string
	Limits         limits.Settings
	// Completion tokens a call is assumed to produce when it doesn't set
	// max_tokens: its budget reservation until the provider reports usage.
	CompletionEstimate int
	Prices             string // GATEWAY_PRICES: JSON on top of the built-in price table

	// The personal data redacted from prompts: kinds from package redact, or
	// none. GATEWAY_REDACT: "all" (the default), "off", or "email,phone,…".
	Redact []redact.Kind

	// The response cache (package cache), on a Redis of its own. Unset: no cache.
	CacheRedisURL string
	CacheTTL      time.Duration
}

// The default daily budget per user: about 20 questions at the ~16,000 input
// tokens one measured on average (README, "End-to-end answers").
const defaultUserBudget = 20 * 16_000

// FromEnv reads the configuration, refusing to start without the keys.
func FromEnv() (Config, error) {
	c := Config{
		Addr:              env("GATEWAY_ADDR", ":8080"),
		MetricsAddr:       env("GATEWAY_METRICS_ADDR", ":9090"),
		Key:               strings.TrimSpace(os.Getenv("LLM_GATEWAY_KEY")),
		OpenAIKey:         strings.TrimSpace(os.Getenv("OPENAI_API_KEY")),
		OpenAIBaseURL:     strings.TrimRight(env("OPENAI_BASE_URL", "https://api.openai.com/v1"), "/"),
		OpenRouterKey:     strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY")),
		OpenRouterBaseURL: strings.TrimRight(env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), "/"),
		BackupJSONSchema:  boolean("GATEWAY_BACKUP_JSON_SCHEMA", true),
		BreakerFailures:   integer("GATEWAY_BREAKER_FAILURES", 5),
		BreakerCooldown:   duration("GATEWAY_BREAKER_COOLDOWN", 30*time.Second),
		ChatTimeout:       duration("GATEWAY_CHAT_TIMEOUT", 180*time.Second),
		StreamTTFBTimeout: duration("GATEWAY_STREAM_TTFB_TIMEOUT", 30*time.Second),
		EmbedTimeout:      duration("GATEWAY_EMBEDDINGS_TIMEOUT", 60*time.Second),
		MaxBodyBytes:      int64(integer("GATEWAY_MAX_BODY_BYTES", 16<<20)),

		LimitsRedisURL: strings.TrimSpace(os.Getenv("GATEWAY_LIMITS_REDIS_URL")),
		Limits: limits.Settings{
			// Rates are set per minute and kept per second
			ConsumerRate:  float64(integer("GATEWAY_RATE_PER_CONSUMER", 60)) / 60,
			ConsumerBurst: float64(integer("GATEWAY_BURST_PER_CONSUMER", 20)),
			GlobalRate:    float64(integer("GATEWAY_RATE_GLOBAL", 300)) / 60,
			GlobalBurst:   float64(integer("GATEWAY_BURST_GLOBAL", 60)),
			UserBudget:    int64(integer("GATEWAY_DAILY_BUDGET_USER", defaultUserBudget)),
			SystemBudget:  int64(integer("GATEWAY_DAILY_BUDGET_SYSTEM", 0)),
			Replicas:      integer("GATEWAY_REPLICAS", 3),
		},
		CompletionEstimate: integer("GATEWAY_COMPLETION_ESTIMATE", 512),
		Prices:             os.Getenv("GATEWAY_PRICES"),

		CacheRedisURL: strings.TrimSpace(os.Getenv("GATEWAY_CACHE_REDIS_URL")),
		CacheTTL:      duration("GATEWAY_CACHE_TTL", 7*24*time.Hour),
	}
	var errs []error
	models, err := parseModels(env("GATEWAY_BACKUP_MODELS", "gpt-4o-mini=openai/gpt-4o-mini"))
	if err != nil {
		errs = append(errs, err)
	}
	c.BackupModels = models
	kinds, err := parseRedact(env("GATEWAY_REDACT", "all"))
	if err != nil {
		errs = append(errs, err)
	}
	c.Redact = kinds
	if len(c.Key) < 16 {
		errs = append(errs, errors.New("LLM_GATEWAY_KEY must be set (16 characters or more)"))
	}
	if c.OpenAIKey == "" {
		errs = append(errs, errors.New("OPENAI_API_KEY must be set"))
	}
	if (c.Limits.ConsumerRate > 0 && c.Limits.ConsumerBurst < 1) ||
		(c.Limits.GlobalRate > 0 && c.Limits.GlobalBurst < 1) {
		errs = append(errs, errors.New("a rate limit needs a burst of 1 or more"))
	}
	return c, errors.Join(errs...)
}

// parseModels reads "asked=backup,asked=backup". An embedding model may only
// map to the same model under another name: vectors from a different model
// can't be compared with the ones already stored.
func parseModels(spec string) (map[string]string, error) {
	out := map[string]string{}
	for _, pair := range strings.Split(spec, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		asked, backup, ok := strings.Cut(pair, "=")
		asked, backup = strings.TrimSpace(asked), strings.TrimSpace(backup)
		if !ok || asked == "" || backup == "" {
			return nil, fmt.Errorf("GATEWAY_BACKUP_MODELS: %q isn't asked=backup", pair)
		}
		if strings.HasPrefix(asked, "text-embedding") && backup != asked && !strings.HasSuffix(backup, "/"+asked) {
			return nil, fmt.Errorf("GATEWAY_BACKUP_MODELS: %s may only fail over to the same model, not %s", asked, backup)
		}
		out[asked] = backup
	}
	return out, nil
}

func parseRedact(spec string) ([]redact.Kind, error) {
	switch strings.ToLower(strings.TrimSpace(spec)) {
	case "all":
		return redact.Kinds, nil
	case "off", "none", "0", "false":
		return nil, nil
	}
	var out []redact.Kind
	for _, name := range strings.Split(spec, ",") {
		kind := redact.Kind(strings.ToUpper(strings.TrimSpace(name)))
		if !slices.Contains(redact.Kinds, kind) {
			return nil, fmt.Errorf("GATEWAY_REDACT: unknown kind %q", name)
		}
		out = append(out, kind)
	}
	return out, nil
}

func boolean(name string, fallback bool) bool {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ignoring %s=%q: not true or false\n", name, v)
		return fallback
	}
	return b
}

func env(name, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return fallback
}

func duration(name string, fallback time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		fmt.Fprintf(os.Stderr, "ignoring %s=%q: not a positive duration\n", name, v)
		return fallback
	}
	return d
}

func integer(name string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		fmt.Fprintf(os.Stderr, "ignoring %s=%q: not a whole number\n", name, v)
		return fallback
	}
	return n
}

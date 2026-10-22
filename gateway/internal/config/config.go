// Package config reads the gateway's settings from the environment.
package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config is everything the gateway is told at start-up.
type Config struct {
	Addr        string // the OpenAI-compatible API
	MetricsAddr string // /metrics, internal only

	// The bearer key every caller must present. The app's LLM_GATEWAY_KEY.
	Key string

	OpenAIKey     string
	OpenAIBaseURL string

	// How long a call may wait for the provider's first byte. Non-streamed
	// chat calls only send theirs when the whole answer is written, and figure
	// extraction can take minutes, so they get the most.
	ChatTimeout       time.Duration
	StreamTTFBTimeout time.Duration
	EmbedTimeout      time.Duration

	MaxBodyBytes int64
}

// FromEnv reads the configuration, refusing to start without the keys.
func FromEnv() (Config, error) {
	c := Config{
		Addr:              env("GATEWAY_ADDR", ":8080"),
		MetricsAddr:       env("GATEWAY_METRICS_ADDR", ":9090"),
		Key:               strings.TrimSpace(os.Getenv("LLM_GATEWAY_KEY")),
		OpenAIKey:         strings.TrimSpace(os.Getenv("OPENAI_API_KEY")),
		OpenAIBaseURL:     strings.TrimRight(env("OPENAI_BASE_URL", "https://api.openai.com/v1"), "/"),
		ChatTimeout:       duration("GATEWAY_CHAT_TIMEOUT", 180*time.Second),
		StreamTTFBTimeout: duration("GATEWAY_STREAM_TTFB_TIMEOUT", 30*time.Second),
		EmbedTimeout:      duration("GATEWAY_EMBEDDINGS_TIMEOUT", 60*time.Second),
		MaxBodyBytes:      int64(integer("GATEWAY_MAX_BODY_BYTES", 16<<20)),
	}
	var errs []error
	if len(c.Key) < 16 {
		errs = append(errs, errors.New("LLM_GATEWAY_KEY must be set (16 characters or more)"))
	}
	if c.OpenAIKey == "" {
		errs = append(errs, errors.New("OPENAI_API_KEY must be set"))
	}
	return c, errors.Join(errs...)
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

// Package cache answers repeated chat calls from Redis instead of the provider.
//
// Exact matches only, never "semantically similar" ones: a question about 2023
// and the same question about 2024 embed almost identically, and a similarity
// cache would answer one with the other's figures. The key is the SHA-256 of
// the canonical JSON of every request field that can change the answer (model,
// messages, temperature, tools, response_format, max_tokens, seed, …), so two
// calls share an entry only if the provider would have been asked exactly the
// same thing. Only temperature-0, single-answer calls are cached: anything
// else is meant to vary.
//
// What is stored is the response body alone, a chat.completion JSON, and only
// for answers that finished normally ("stop" or "tool_calls"): never errors,
// cut-off streams or truncated answers. A streamed call is assembled into that
// JSON before it is stored, and a streamed hit is replayed as SSE from it.
package cache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// Fields that change how an answer is delivered or recorded, not what it says
var ignored = map[string]bool{
	"stream": true, "stream_options": true, "user": true, "metadata": true, "store": true,
	"service_tier": true, "prompt_cache_key": true, "safety_identifier": true,
}

// MaxEntry is the largest body stored; bigger answers are left uncached.
const MaxEntry = 1 << 20

// Key is the cache key for a chat request's fields, and whether it may be
// cached at all (temperature 0, one answer).
func Key(fields map[string]json.RawMessage) (string, bool) {
	if !zero(fields["temperature"]) {
		return "", false
	}
	if raw, ok := fields["n"]; ok {
		var n int
		if json.Unmarshal(raw, &n) != nil || n != 1 {
			return "", false
		}
	}
	canonical := map[string]any{}
	for name, raw := range fields {
		if ignored[name] || name == "n" { // n is 1 by now: the default, as if absent
			continue
		}
		value, err := decode(raw)
		if err != nil {
			return "", false
		}
		canonical[name] = value
	}
	// encoding/json writes map keys sorted, without spaces: one spelling per request
	out, err := json.Marshal(canonical)
	if err != nil {
		return "", false
	}
	sum := sha256.Sum256(out)
	return "gw:cache:v1:" + hex.EncodeToString(sum[:]), true
}

// decode parses JSON into a canonical value: numbers in one spelling each, so
// 0, 0.0 and 0e0 make the same key.
func decode(raw json.RawMessage) (any, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var v any
	if err := d.Decode(&v); err != nil {
		return nil, err
	}
	return canonical(v), nil
}

func canonical(v any) any {
	switch x := v.(type) {
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return json.Number(strconv.FormatInt(i, 10))
		}
		if f, err := x.Float64(); err == nil {
			if f == float64(int64(f)) && f < 1<<53 && f > -(1<<53) {
				return json.Number(strconv.FormatInt(int64(f), 10))
			}
			return json.Number(strconv.FormatFloat(f, 'g', -1, 64))
		}
		return x
	case map[string]any:
		for k, item := range x {
			x[k] = canonical(item)
		}
	case []any:
		for i, item := range x {
			x[i] = canonical(item)
		}
	}
	return v
}

func zero(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false // absent means the provider's default of 1
	}
	f, err := strconv.ParseFloat(string(bytes.TrimSpace(raw)), 64)
	return err == nil && f == 0
}

// Cache is the response cache on its own Redis (allkeys-lru, so it evicts the
// oldest entries when full; the limits' Redis never evicts).
type Cache struct {
	rdb redis.UniversalClient
	ttl time.Duration
}

// New returns a cache on rdb whose entries live for ttl.
func New(rdb redis.UniversalClient, ttl time.Duration) *Cache {
	return &Cache{rdb: rdb, ttl: ttl}
}

// Get returns the stored body for key. A Redis error is a miss: the cache may
// fail, the call never does because of it.
func (c *Cache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 150*time.Millisecond)
	defer cancel()
	body, err := c.rdb.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return body, true, nil
}

// Put stores a finished answer's body, if it is one worth storing.
func (c *Cache) Put(ctx context.Context, key string, body []byte) error {
	if len(body) > MaxEntry || !Finished(body) {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 250*time.Millisecond)
	defer cancel()
	return c.rdb.Set(ctx, key, body, c.ttl).Err()
}

// Completion is the part of a chat.completion the cache reads.
type Completion struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	SystemFingerprint *string         `json:"system_fingerprint,omitempty"`
	Choices           []Choice        `json:"choices"`
	Usage             json.RawMessage `json:"usage,omitempty"`
}

// Choice is one answer.
type Choice struct {
	Index        int             `json:"index"`
	Message      json.RawMessage `json:"message"`
	FinishReason *string         `json:"finish_reason"`
	Logprobs     json.RawMessage `json:"logprobs,omitempty"`
}

// Finished is true for a single answer that ended normally.
func Finished(body []byte) bool {
	var c Completion
	if json.Unmarshal(body, &c) != nil || len(c.Choices) != 1 || c.Choices[0].FinishReason == nil {
		return false
	}
	switch *c.Choices[0].FinishReason {
	case "stop", "tool_calls":
		return true
	}
	return false
}

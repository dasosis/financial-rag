// Package fakeupstream is a stand-in for the OpenAI API: chat completions
// (plain, streamed, and structured output from a JSON schema) and embeddings,
// answered deterministically and offline. The gateway's tests, the Python
// integration test and the load test all run against it, so none of them spends
// real money.
//
// Tests steer it through Options (latency before the first byte, delay between
// streamed chunks, the model reported) and Fail / Truncate (errors, and streams
// that stop mid-answer).
package fakeupstream

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Options steer the fake. The zero value answers at once.
type Options struct {
	Latency    time.Duration // before the response starts
	ChunkDelay time.Duration // between streamed chunks
	// Model, if set, is reported instead of the model asked for
	Model string
	// Dimensions of each embedding (default 1536, as text-embedding-3-small)
	Dimensions int
}

// Fake is an http.Handler serving /v1/chat/completions and /v1/embeddings.
type Fake struct {
	mu   sync.Mutex
	opts Options
	// fail, when non-zero, is the status every request gets
	fail      atomic.Int32
	failCode  atomic.Value // string
	truncate  atomic.Bool  // end streams without a finish_reason or [DONE]
	calls     atomic.Int64
	lastMu    sync.Mutex
	last      map[string]any // the last request body, for assertions
	lastHeads http.Header
}

// New returns a fake with the given options.
func New(opts Options) *Fake {
	if opts.Dimensions == 0 {
		opts.Dimensions = 1536
	}
	return &Fake{opts: opts}
}

// SetOptions replaces the options.
func (f *Fake) SetOptions(opts Options) {
	if opts.Dimensions == 0 {
		opts.Dimensions = 1536
	}
	f.mu.Lock()
	f.opts = opts
	f.mu.Unlock()
}

func (f *Fake) options() Options {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.opts
}

// Fail makes every request answer with status and an OpenAI-style error with
// this code; Fail(0, "") goes back to answering normally.
func (f *Fake) Fail(status int, code string) {
	f.fail.Store(int32(status))
	f.failCode.Store(code)
}

// Truncate makes streams stop before their finish_reason and [DONE].
func (f *Fake) Truncate(on bool) { f.truncate.Store(on) }

// Calls is how many requests the fake has answered.
func (f *Fake) Calls() int64 { return f.calls.Load() }

// Last is the body of the most recent request, decoded.
func (f *Fake) Last() map[string]any {
	f.lastMu.Lock()
	defer f.lastMu.Unlock()
	return f.last
}

// LastHeaders are the headers of the most recent request.
func (f *Fake) LastHeaders() http.Header {
	f.lastMu.Lock()
	defer f.lastMu.Unlock()
	return f.lastHeads
}

func (f *Fake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.calls.Add(1)
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}
	f.lastMu.Lock()
	f.last, f.lastHeads = body, r.Header.Clone()
	f.lastMu.Unlock()

	opts := f.options()
	if opts.Latency > 0 {
		select {
		case <-time.After(opts.Latency):
		case <-r.Context().Done():
			return
		}
	}
	if status := int(f.fail.Load()); status != 0 {
		code, _ := f.failCode.Load().(string)
		writeError(w, status, code, "fake failure")
		return
	}
	switch {
	case strings.HasSuffix(r.URL.Path, "/chat/completions"):
		f.chat(w, r, body, opts)
	case strings.HasSuffix(r.URL.Path, "/embeddings"):
		f.embeddings(w, body, opts)
	default:
		writeError(w, http.StatusNotFound, "not_found", "no such endpoint")
	}
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "fake_error", "code": code},
	})
}

// Answer is the reply to a conversation: a fixed sentence naming the last
// message's length, so different prompts get different (but repeatable) answers.
func Answer(body map[string]any) string {
	last := ""
	if msgs, ok := body["messages"].([]any); ok && len(msgs) > 0 {
		if m, ok := msgs[len(msgs)-1].(map[string]any); ok {
			last = contentText(m["content"])
		}
	}
	sum := sha256.Sum256([]byte(last))
	return fmt.Sprintf("Fake answer %x to a message of %d characters.", sum[:4], len(last))
}

func contentText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any: // [{"type": "text", "text": "..."}]
		var parts []string
		for _, p := range c {
			if m, ok := p.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, " ")
	}
	return ""
}

// promptTokens estimates tokens the way the gateway does: about four characters each.
func promptTokens(body map[string]any) int {
	raw, _ := json.Marshal(body["messages"])
	return len(raw)/4 + 1
}

func (f *Fake) content(body map[string]any) string {
	if rf, ok := body["response_format"].(map[string]any); ok && rf["type"] == "json_schema" {
		if js, ok := rf["json_schema"].(map[string]any); ok {
			schema, _ := js["schema"].(map[string]any)
			out, _ := json.Marshal(Instance(schema, schema))
			return string(out)
		}
	}
	if rf, ok := body["response_format"].(map[string]any); ok && rf["type"] == "json_object" {
		return `{}`
	}
	return Answer(body)
}

func (f *Fake) model(body map[string]any, opts Options) string {
	if opts.Model != "" {
		return opts.Model
	}
	m, _ := body["model"].(string)
	return m
}

func (f *Fake) chat(w http.ResponseWriter, r *http.Request, body map[string]any, opts Options) {
	content := f.content(body)
	model := f.model(body, opts)
	prompt := promptTokens(body)
	completion := len(content)/4 + 1
	usage := map[string]any{
		"prompt_tokens": prompt, "completion_tokens": completion,
		"total_tokens":          prompt + completion,
		"prompt_tokens_details": map[string]any{"cached_tokens": 0},
	}
	id := "chatcmpl-fake"
	created := time.Now().Unix()

	if stream, _ := body["stream"].(bool); !stream {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Request-Id", "req-fake")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": model,
			"choices": []any{map[string]any{
				"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": content, "refusal": nil},
			}},
			"usage": usage,
		})
		return
	}

	wantUsage := false
	if so, ok := body["stream_options"].(map[string]any); ok {
		wantUsage, _ = so["include_usage"].(bool)
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	send := func(chunk map[string]any) bool {
		raw, _ := json.Marshal(chunk)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", raw); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	chunk := func(delta map[string]any, finish any) map[string]any {
		c := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		}
		if wantUsage {
			c["usage"] = nil
		}
		return c
	}
	if !send(chunk(map[string]any{"role": "assistant", "content": ""}, nil)) {
		return
	}
	words := strings.SplitAfter(content, " ")
	for i, word := range words {
		if opts.ChunkDelay > 0 {
			select {
			case <-time.After(opts.ChunkDelay):
			case <-r.Context().Done():
				return
			}
		}
		if !send(chunk(map[string]any{"content": word}, nil)) {
			return
		}
		if f.truncate.Load() && i >= len(words)/2 {
			return // the connection drops mid-answer
		}
	}
	if !send(chunk(map[string]any{}, "stop")) {
		return
	}
	if wantUsage {
		send(map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{}, "usage": usage,
		})
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func (f *Fake) embeddings(w http.ResponseWriter, body map[string]any, opts Options) {
	var inputs []string
	switch in := body["input"].(type) {
	case string:
		inputs = []string{in}
	case []any:
		for _, item := range in {
			switch v := item.(type) {
			case string:
				inputs = append(inputs, v)
			default: // token ids: embed their JSON
				raw, _ := json.Marshal(v)
				inputs = append(inputs, string(raw))
			}
		}
	}
	data := make([]any, len(inputs))
	tokens := 0
	for i, text := range inputs {
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": Vector(text, opts.Dimensions)}
		tokens += len(text)/4 + 1
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list", "data": data, "model": f.model(body, opts),
		"usage": map[string]any{"prompt_tokens": tokens, "total_tokens": tokens},
	})
}

// Vector is a deterministic unit vector for text: the same text, the same vector.
func Vector(text string, dims int) []float64 {
	out := make([]float64, dims)
	seed := sha256.Sum256([]byte(text))
	var norm float64
	for i := range out {
		if i%8 == 0 && i > 0 {
			seed = sha256.Sum256(seed[:])
		}
		v := float64(int32(binary.LittleEndian.Uint32(seed[(i%8)*4:])))/math.MaxInt32 + 1e-9
		out[i] = v
		norm += v * v
	}
	norm = math.Sqrt(norm)
	for i := range out {
		out[i] /= norm
	}
	return out
}

// Instance is a minimal value that satisfies a JSON schema: every property
// present, strings "fake", numbers 1, arrays empty. $ref into $defs is followed.
func Instance(schema, root map[string]any) any {
	if ref, ok := schema["$ref"].(string); ok {
		name := ref[strings.LastIndex(ref, "/")+1:]
		for _, key := range []string{"$defs", "definitions"} {
			if defs, ok := root[key].(map[string]any); ok {
				if def, ok := defs[name].(map[string]any); ok {
					return Instance(def, root)
				}
			}
		}
		return nil
	}
	if anyOf, ok := schema["anyOf"].([]any); ok && len(anyOf) > 0 {
		if first, ok := anyOf[0].(map[string]any); ok {
			return Instance(first, root)
		}
	}
	if enum, ok := schema["enum"].([]any); ok && len(enum) > 0 {
		return enum[0]
	}
	typ := schema["type"]
	if types, ok := typ.([]any); ok && len(types) > 0 {
		typ = types[0]
	}
	switch typ {
	case "object":
		out := map[string]any{}
		props, _ := schema["properties"].(map[string]any)
		for name, p := range props {
			if ps, ok := p.(map[string]any); ok {
				out[name] = Instance(ps, root)
			}
		}
		return out
	case "array":
		return []any{}
	case "string":
		return "fake"
	case "integer", "number":
		return 1
	case "boolean":
		return true
	case "null":
		return nil
	}
	return nil
}

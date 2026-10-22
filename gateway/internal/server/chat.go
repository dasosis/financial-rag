package server

import (
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/cache"
	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

// chatRequest is a chat completions body, decoded only as far as the gateway
// needs. Every other field is passed through as the caller sent it.
type chatRequest struct {
	fields map[string]json.RawMessage
	model  string
	stream bool
	// The caller asked for the usage chunk at the end of a stream. The gateway
	// always asks for it (it is where token counts come from) and drops it again
	// for callers who didn't.
	clientUsage bool
	// The schema of response_format json_schema (structured output), if any
	schema json.RawMessage
}

var errBadRequest = errors.New("bad request")

func parseChat(body []byte) (*chatRequest, error) {
	req := &chatRequest{}
	if err := json.Unmarshal(body, &req.fields); err != nil || req.fields == nil {
		return nil, errBadRequest
	}
	if err := json.Unmarshal(req.fields["model"], &req.model); err != nil || req.model == "" {
		return nil, errBadRequest
	}
	if raw, ok := req.fields["stream"]; ok {
		if err := json.Unmarshal(raw, &req.stream); err != nil {
			return nil, errBadRequest
		}
	}
	var format struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if raw, ok := req.fields["response_format"]; ok && json.Unmarshal(raw, &format) == nil &&
		format.Type == "json_schema" {
		req.schema = format.JSONSchema.Schema
		if len(req.schema) == 0 {
			req.schema = json.RawMessage("{}")
		}
	}
	if !req.stream {
		return req, nil
	}
	options := map[string]json.RawMessage{}
	if raw, ok := req.fields["stream_options"]; ok && string(raw) != "null" {
		if err := json.Unmarshal(raw, &options); err != nil {
			return nil, errBadRequest
		}
	}
	if raw, ok := options["include_usage"]; ok {
		_ = json.Unmarshal(raw, &req.clientUsage)
	}
	options["include_usage"] = json.RawMessage("true")
	req.fields["stream_options"], _ = json.Marshal(options)
	return req, nil
}

// plan is how the call goes to one provider: under that provider's name for
// the model. A backup answering a structured-output call is asked for its
// answer whole, even if the caller streams, so the answer can be checked
// against the schema before any of it goes out (and replayed as SSE after).
func (r *chatRequest) plan(p *Provider) (outbound, bool) {
	model, ok := p.Model(r.model)
	if !ok {
		return outbound{}, false
	}
	checked := r.schema != nil && !p.Primary()
	if checked && !p.JSONSchema {
		return outbound{}, false
	}
	fields := maps.Clone(r.fields)
	fields["model"], _ = json.Marshal(model)
	stream := r.stream
	if checked {
		delete(fields, "stream")
		delete(fields, "stream_options")
		stream = false
	}
	body, _ := json.Marshal(fields)
	return outbound{body: body, stream: stream}, true
}

func (s *Server) chat(c *call) {
	body, ok := s.readBody(c)
	if !ok {
		return
	}
	req, err := parseChat(body)
	if err != nil {
		s.fail(c, http.StatusBadRequest, "invalid_request_error", "invalid_body",
			"The body must be a JSON chat completions request with a model.")
		return
	}
	lookup := s.cacheLookup(c, req)
	if lookup.hit != nil {
		s.serveCached(c, req, lookup.hit)
		return
	}
	reservation, ok := s.admit(c, req.model, req.estimate(s.cfg.CompletionEstimate))
	if !ok {
		return
	}
	resp, p, out, ok := s.dispatch(c, "/chat/completions", req.plan)
	if !ok {
		s.settle(c, reservation, 0) // nothing was generated
		return
	}
	defer resp.Close()
	if resp.StatusCode != http.StatusOK {
		s.relayError(c, resp)
		s.settle(c, reservation, 0)
		return
	}
	// A forced call (a benchmark scoring the backup) stays out of the cache
	store := lookup.store && c.r.Header.Get("X-Force-Provider") == ""

	if out.stream {
		var assembler *cache.Assembler
		if store {
			assembler = &cache.Assembler{}
		}
		result := s.relayStream(c, resp, req.clientUsage, assembler)
		s.settle(c, reservation, s.account(req.model, result.usage))
		s.served(p, c)
		if assembler != nil && result.complete() {
			if completion, ok := assembler.Completion(); ok {
				s.cacheStore(c, lookup.key, completion)
			}
		}
		s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint, "stream", true,
			"provider", p.Name, "model", req.model, "finished", result.finish,
			"complete", result.complete(), "ms", time.Since(c.start).Milliseconds())
		return
	}

	answer, used, err := readAnswer(resp)
	if err != nil {
		p.Breaker.Failure()
		s.settle(c, reservation, -1)
		s.fail(c, http.StatusServiceUnavailable, "api_error", CodeProvidersUnavailable,
			"The model provider's answer broke off.")
		return
	}
	s.settle(c, reservation, s.account(req.model, used))
	if req.schema != nil && !p.Primary() {
		if err := checkStructured(answer, req.schema); err != nil {
			// Never pass on a malformed structured answer: the app would fail to
			// parse it, or worse, act on it. 503, as if no provider could answer.
			p.Breaker.Failure()
			s.log.Warn("backup's structured output didn't match the schema",
				"request_id", c.requestID, "provider", p.Name, "error", err.Error())
			s.fail(c, http.StatusServiceUnavailable, "api_error", CodeProvidersUnavailable,
				"The backup provider couldn't produce the structured output asked for.")
			return
		}
	}
	if req.stream { // asked whole from a backup to check it: replay it as the stream asked for
		events, err := cache.Replay(answer, req.clientUsage)
		if err != nil {
			s.fail(c, http.StatusServiceUnavailable, "api_error", CodeProvidersUnavailable,
				"The model provider's answer couldn't be streamed.")
			return
		}
		s.writeSSE(c, events)
	} else {
		copyHeaders(c.w.Header(), resp.Header, "Content-Type", "x-request-id")
		c.w.WriteHeader(http.StatusOK)
		_, _ = c.w.Write(answer)
	}
	s.served(p, c)
	if store {
		s.cacheStore(c, lookup.key, answer)
	}
	s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint, "stream", req.stream,
		"provider", p.Name, "model", req.model, "ms", time.Since(c.start).Milliseconds())
}

// estimate is the call's likely usage before it is made: its prompt at about
// four characters a token, and the answer at max_tokens if set, or `answer`.
func (r *chatRequest) estimate(answer int) usage.Usage {
	prompt := (len(r.fields["messages"]) + len(r.fields["tools"]) + len(r.fields["response_format"])) / 4
	for _, name := range []string{"max_completion_tokens", "max_tokens"} {
		var n int
		if raw, ok := r.fields[name]; ok && json.Unmarshal(raw, &n) == nil && n > 0 {
			answer = n
			break
		}
	}
	return usage.Usage{PromptTokens: prompt + 1, CompletionTokens: answer}
}

// readBody reads the request body, refusing one over the size limit.
func (s *Server) readBody(c *call) ([]byte, bool) {
	// A deadline for the body only: a slow sender can't hold the replica, and
	// the answer, which may stream for minutes, isn't affected.
	rc := http.NewResponseController(c.w)
	_ = rc.SetReadDeadline(time.Now().Add(30 * time.Second))
	body, err := io.ReadAll(io.LimitReader(c.r.Body, s.cfg.MaxBodyBytes+1))
	_ = rc.SetReadDeadline(time.Time{})
	if err != nil {
		s.fail(c, http.StatusBadRequest, "invalid_request_error", "invalid_body", "Couldn't read the body.")
		return nil, false
	}
	if int64(len(body)) > s.cfg.MaxBodyBytes {
		s.fail(c, http.StatusRequestEntityTooLarge, "invalid_request_error", "body_too_large",
			"The request body is too large.")
		return nil, false
	}
	return body, true
}

// served counts a response a provider gave, by provider.
func (s *Server) served(p *Provider, c *call) {
	s.metrics.Served.WithLabelValues(p.Name, c.endpoint).Inc()
}

// relayError passes a provider's error response (a 4xx other than 429) on as it came.
func (s *Server) relayError(c *call, resp *upstreamResponse) {
	body, _ := io.ReadAll(io.LimitReader(resp.body, 1<<20))
	copyHeaders(c.w.Header(), resp.Header, "Content-Type", "Retry-After", "x-request-id")
	c.w.WriteHeader(resp.StatusCode)
	_, _ = c.w.Write(body)
	s.log.Info("provider error", "request_id", c.requestID, "endpoint", c.endpoint,
		"status", resp.StatusCode)
}

// readAnswer reads a provider's JSON answer and the usage in it.
func readAnswer(resp *upstreamResponse) ([]byte, usage.Usage, error) {
	body, err := io.ReadAll(io.LimitReader(resp.body, 64<<20))
	if err != nil {
		return nil, usage.Usage{}, err
	}
	var parsed struct {
		Usage *usage.Usage `json:"usage"`
	}
	_ = json.Unmarshal(body, &parsed)
	if parsed.Usage == nil {
		return body, usage.Usage{}, nil
	}
	return body, *parsed.Usage, nil
}

func copyHeaders(dst, src http.Header, names ...string) {
	for _, name := range names {
		if v := src.Get(name); v != "" {
			dst.Set(name, v)
		}
	}
}

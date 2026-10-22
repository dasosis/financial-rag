package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
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

func (r *chatRequest) encode() []byte {
	out, _ := json.Marshal(r.fields)
	return out
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
	firstByte := s.cfg.ChatTimeout
	if req.stream {
		firstByte = s.cfg.StreamTTFBTimeout
	}
	resp, err := s.forward(c, "/chat/completions", req.encode(), firstByte)
	if err != nil {
		s.settle(c, reservation, 0) // nothing was generated
		return
	}
	defer resp.Close()
	if resp.StatusCode != http.StatusOK {
		s.relayError(c, resp)
		s.settle(c, reservation, 0)
		return
	}
	if req.stream {
		var assembler *cache.Assembler
		if lookup.store {
			assembler = &cache.Assembler{}
		}
		result := s.relayStream(c, resp, req.clientUsage, assembler)
		s.settle(c, reservation, s.account(req.model, result.usage))
		if assembler != nil && result.complete() {
			if completion, ok := assembler.Completion(); ok {
				s.cacheStore(c, lookup.key, completion)
			}
		}
		s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint, "stream", true,
			"model", req.model, "finished", result.finish, "complete", result.complete(),
			"ms", time.Since(c.start).Milliseconds())
		return
	}
	answer, used := s.relayJSON(c, resp)
	s.settle(c, reservation, s.account(req.model, used))
	if lookup.store && answer != nil {
		s.cacheStore(c, lookup.key, answer)
	}
	s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint, "stream", false,
		"model", req.model, "ms", time.Since(c.start).Milliseconds())
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

// forward sends the call to the provider. On failure it has already answered
// the caller (and returns the error only so the handler stops).
func (s *Server) forward(c *call, path string, body []byte, firstByte time.Duration) (*upstreamResponse, error) {
	p := s.provider
	sent := time.Now()
	resp, err := p.send(c.r.Context(), path, body, firstByte)
	if err != nil {
		s.metrics.Upstream.WithLabelValues(p.Name, c.endpoint, "error").Inc()
		s.upstreamFailed(c, err)
		return nil, err
	}
	s.metrics.UpLatency.WithLabelValues(p.Name, c.endpoint).Observe(time.Since(sent).Seconds())
	s.metrics.Upstream.WithLabelValues(p.Name, c.endpoint, strconv.Itoa(resp.StatusCode)).Inc()
	return resp, nil
}

// upstreamFailed answers a call whose provider couldn't be reached. These are
// 502/504 without x-should-retry, so the app's client retries them as it
// would a failed call to OpenAI itself.
func (s *Server) upstreamFailed(c *call, err error) {
	switch {
	case c.r.Context().Err() != nil:
		// The caller hung up; there is no one to answer. 499 as nginx logs it.
		c.w.code = 499
	case errors.Is(err, errFirstByteTimeout), errors.Is(err, context.DeadlineExceeded):
		s.fail(c, http.StatusGatewayTimeout, "api_error", "upstream_timeout",
			"The model provider didn't answer in time.")
	default:
		s.log.Warn("provider unreachable", "request_id", c.requestID, "error", err.Error())
		s.fail(c, http.StatusBadGateway, "api_error", "upstream_unavailable",
			"Couldn't reach the model provider.")
	}
}

// relayError passes a provider's error response on as it came.
func (s *Server) relayError(c *call, resp *upstreamResponse) {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	copyHeaders(c.w.Header(), resp.Header, "Content-Type", "Retry-After", "x-request-id")
	c.w.WriteHeader(resp.StatusCode)
	_, _ = c.w.Write(body)
	s.log.Info("provider error", "request_id", c.requestID, "endpoint", c.endpoint,
		"status", resp.StatusCode)
}

// relayJSON passes a provider's JSON answer on, and returns it and its usage.
func (s *Server) relayJSON(c *call, resp *upstreamResponse) ([]byte, usage.Usage) {
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		s.upstreamFailed(c, err)
		return nil, usage.Usage{}
	}
	var parsed struct {
		Usage *usage.Usage `json:"usage"`
	}
	_ = json.Unmarshal(body, &parsed)
	copyHeaders(c.w.Header(), resp.Header, "Content-Type", "x-request-id")
	c.w.WriteHeader(http.StatusOK)
	_, _ = c.w.Write(body)
	if parsed.Usage == nil {
		return body, usage.Usage{}
	}
	return body, *parsed.Usage
}

func copyHeaders(dst, src http.Header, names ...string) {
	for _, name := range names {
		if v := src.Get(name); v != "" {
			dst.Set(name, v)
		}
	}
}

// streamResult is what a relayed stream turned out to be.
type streamResult struct {
	usage  usage.Usage
	finish string // the last finish_reason seen: "stop", "length", "tool_calls"…
	done   bool   // "data: [DONE]" arrived
	err    error  // reading the provider or writing to the caller failed
}

// complete is true for a stream that ended normally: a finish reason, [DONE],
// and nothing broken on the way.
func (r streamResult) complete() bool {
	return r.err == nil && r.done && r.finish != ""
}

// relayStream copies a provider's SSE stream to the caller event by event,
// flushing each at once (never buffering the answer). The usage chunk is read
// for its token counts and dropped unless the caller asked for it.
//
// A non-nil assembler is given every chunk, to rebuild the answer for the cache.
func (s *Server) relayStream(c *call, resp *upstreamResponse, clientUsage bool, assembler *cache.Assembler) streamResult {
	h := c.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	copyHeaders(h, resp.Header, "x-request-id")
	c.w.WriteHeader(http.StatusOK)
	c.w.Flush()

	var result streamResult
	reader := bufio.NewReaderSize(resp.Body, 64<<10)
	for {
		event, err := readEvent(reader)
		if len(event) > 0 {
			keep := true
			if data, ok := eventData(event); ok {
				keep = inspectChunk(data, &result, clientUsage)
				if assembler != nil && string(bytes.TrimSpace(data)) != "[DONE]" {
					assembler.Add(data)
				}
			}
			if keep {
				if _, werr := c.w.Write(event); werr != nil {
					result.err = werr
					return result
				}
				c.w.Flush()
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				result.err = err
			} else if !result.done {
				result.err = io.ErrUnexpectedEOF
			}
			return result
		}
	}
}

// readEvent reads one SSE event: lines up to and including a blank line.
func readEvent(r *bufio.Reader) ([]byte, error) {
	var event []byte
	for {
		line, err := r.ReadBytes('\n')
		event = append(event, line...)
		if err != nil {
			return event, err
		}
		if len(bytes.TrimRight(line, "\r\n")) == 0 {
			return event, nil
		}
	}
}

// eventData is the payload of an event's data lines, joined.
func eventData(event []byte) ([]byte, bool) {
	var data [][]byte
	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimRight(line, "\r")
		if rest, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			data = append(data, bytes.TrimPrefix(rest, []byte(" ")))
		}
	}
	if len(data) == 0 {
		return nil, false
	}
	return bytes.Join(data, []byte("\n")), true
}

// inspectChunk notes what a chunk says (usage, finish reason, the end) and
// says whether to pass it on.
func inspectChunk(data []byte, result *streamResult, clientUsage bool) bool {
	if string(bytes.TrimSpace(data)) == "[DONE]" {
		result.done = true
		return true
	}
	var chunk struct {
		Choices []struct {
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usage.Usage `json:"usage"`
	}
	if json.Unmarshal(data, &chunk) != nil {
		return true
	}
	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			result.finish = *choice.FinishReason
		}
	}
	if chunk.Usage != nil {
		result.usage = *chunk.Usage
		if len(chunk.Choices) == 0 && !clientUsage {
			return false
		}
	}
	return true
}

package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/dasosis/financial-rag/gateway/internal/cache"
	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

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
	reader := bufio.NewReaderSize(resp.body, 64<<10)
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

// writeSSE sends a whole stream of events at once (a replayed answer).
func (s *Server) writeSSE(c *call, events []byte) {
	h := c.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	c.w.WriteHeader(http.StatusOK)
	_, _ = c.w.Write(events)
	c.w.Flush()
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

package cache

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// Assembler rebuilds a streamed answer as the chat.completion JSON it would
// have been unstreamed, so one stored form serves streamed and plain calls.
type Assembler struct {
	id, model   string
	created     int64
	fingerprint *string
	content     strings.Builder
	hasContent  bool
	refusal     strings.Builder
	hasRefusal  bool
	tools       []*toolCall
	finish      *string
	usage       json.RawMessage
	unusual     bool // something the stored form can't hold: don't cache
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chunk struct {
	ID                string  `json:"id"`
	Created           int64   `json:"created"`
	Model             string  `json:"model"`
	SystemFingerprint *string `json:"system_fingerprint"`
	Choices           []struct {
		Index int `json:"index"`
		Delta struct {
			Content   *string `json:"content"`
			Refusal   *string `json:"refusal"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string         `json:"finish_reason"`
		Logprobs     json.RawMessage `json:"logprobs"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
}

// Add takes one chunk's data (not "[DONE]").
func (a *Assembler) Add(data []byte) {
	var c chunk
	if json.Unmarshal(data, &c) != nil {
		a.unusual = true
		return
	}
	if a.id == "" {
		a.id, a.model, a.created, a.fingerprint = c.ID, c.Model, c.Created, c.SystemFingerprint
	}
	if len(c.Usage) > 0 && string(c.Usage) != "null" {
		a.usage = append(json.RawMessage(nil), c.Usage...)
	}
	for _, choice := range c.Choices {
		if choice.Index != 0 || (len(choice.Logprobs) > 0 && string(choice.Logprobs) != "null") {
			a.unusual = true
			continue
		}
		d := choice.Delta
		if d.Content != nil {
			a.content.WriteString(*d.Content)
			a.hasContent = a.hasContent || *d.Content != ""
		}
		if d.Refusal != nil {
			a.refusal.WriteString(*d.Refusal)
			a.hasRefusal = true
		}
		for _, tc := range d.ToolCalls {
			for len(a.tools) <= tc.Index {
				a.tools = append(a.tools, &toolCall{Type: "function"})
			}
			t := a.tools[tc.Index]
			if tc.ID != "" {
				t.ID = tc.ID
			}
			if tc.Type != "" {
				t.Type = tc.Type
			}
			t.Function.Name += tc.Function.Name
			t.Function.Arguments += tc.Function.Arguments
		}
		if choice.FinishReason != nil {
			reason := *choice.FinishReason
			a.finish = &reason
		}
	}
}

// Completion is the assembled answer, or false if it can't be stored.
func (a *Assembler) Completion() ([]byte, bool) {
	if a.unusual || a.finish == nil {
		return nil, false
	}
	message := map[string]any{"role": "assistant", "content": nil, "refusal": nil}
	if a.hasContent || len(a.tools) == 0 {
		message["content"] = a.content.String()
	}
	if a.hasRefusal {
		message["refusal"] = a.refusal.String()
	}
	if len(a.tools) > 0 {
		message["tool_calls"] = a.tools
	}
	out := map[string]any{
		"id": a.id, "object": "chat.completion", "created": a.created, "model": a.model,
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": *a.finish,
			"logprobs": nil}},
	}
	if a.fingerprint != nil {
		out["system_fingerprint"] = *a.fingerprint
	}
	if len(a.usage) > 0 {
		out["usage"] = a.usage
	}
	body, err := json.Marshal(out)
	return body, err == nil
}

// Replay is a stored answer as the SSE stream the provider would have sent:
// the role, the whole text in one chunk, the finish, the usage chunk if the
// caller asked for it, and [DONE].
func Replay(body []byte, includeUsage bool) ([]byte, error) {
	var c Completion
	if err := json.Unmarshal(body, &c); err != nil || len(c.Choices) != 1 {
		return nil, fmt.Errorf("not a stored completion")
	}
	var message struct {
		Content   *string           `json:"content"`
		Refusal   *string           `json:"refusal"`
		ToolCalls []json.RawMessage `json:"tool_calls"`
	}
	if err := json.Unmarshal(c.Choices[0].Message, &message); err != nil {
		return nil, err
	}
	var out bytes.Buffer
	event := func(choices []any, usage any) {
		chunk := map[string]any{"id": c.ID, "object": "chat.completion.chunk", "created": c.Created,
			"model": c.Model, "choices": choices}
		if c.SystemFingerprint != nil {
			chunk["system_fingerprint"] = *c.SystemFingerprint
		}
		if includeUsage {
			chunk["usage"] = usage
		}
		raw, _ := json.Marshal(chunk)
		out.WriteString("data: ")
		out.Write(raw)
		out.WriteString("\n\n")
	}
	delta := func(d map[string]any, finish any) []any {
		return []any{map[string]any{"index": 0, "delta": d, "finish_reason": finish, "logprobs": nil}}
	}
	event(delta(map[string]any{"role": "assistant", "content": "", "refusal": nil}, nil), nil)
	if message.Content != nil && *message.Content != "" {
		event(delta(map[string]any{"content": *message.Content}, nil), nil)
	}
	if message.Refusal != nil {
		event(delta(map[string]any{"refusal": *message.Refusal}, nil), nil)
	}
	if len(message.ToolCalls) > 0 {
		calls := make([]any, len(message.ToolCalls))
		for i, raw := range message.ToolCalls {
			var call map[string]any
			_ = json.Unmarshal(raw, &call)
			call["index"] = i
			calls[i] = call
		}
		event(delta(map[string]any{"tool_calls": calls}, nil), nil)
	}
	event(delta(map[string]any{}, *c.Choices[0].FinishReason), nil)
	if includeUsage && len(c.Usage) > 0 {
		event([]any{}, c.Usage)
	}
	out.WriteString("data: [DONE]\n\n")
	return out.Bytes(), nil
}

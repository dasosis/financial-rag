package server

import (
	"bytes"
	"encoding/json"
)

// redactChat replaces personal data in every message's text (plain content or
// text parts) with placeholders. It runs before the cache key is made and
// before anything goes to a provider; only the counts are recorded.
func (s *Server) redactChat(req *chatRequest) {
	if s.redactor == nil {
		return
	}
	var messages []map[string]json.RawMessage
	if json.Unmarshal(req.fields["messages"], &messages) != nil {
		return
	}
	changed := false
	for _, m := range messages {
		if out, ok := s.redactContent(m["content"]); ok {
			m["content"] = out
			changed = true
		}
	}
	if changed {
		req.fields["messages"] = marshal(messages)
	}
}

// redactContent redacts a message's content: a string, or a list of parts
// whose "text" parts are redacted. ok is false when nothing changed.
func (s *Server) redactContent(raw json.RawMessage) (json.RawMessage, bool) {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		if out, changed := s.redactText(text); changed {
			return marshal(out), true
		}
		return nil, false
	}
	var parts []map[string]json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return nil, false
	}
	changed := false
	for _, part := range parts {
		var t string
		if json.Unmarshal(part["text"], &t) != nil {
			continue
		}
		if out, ok := s.redactText(t); ok {
			part["text"] = marshal(out)
			changed = true
		}
	}
	if !changed {
		return nil, false
	}
	return marshal(parts), true
}

// redactEmbeddings redacts an embeddings body's input: a string or a list of
// strings. Token-id input can't be read, and is passed on as it is (the app
// sends text: config.embeddings).
func (s *Server) redactEmbeddings(body []byte, fields map[string]json.RawMessage) ([]byte, map[string]json.RawMessage) {
	if s.redactor == nil {
		return body, fields
	}
	raw := fields["input"]
	var one string
	var many []string
	switch {
	case json.Unmarshal(raw, &one) == nil:
		out, changed := s.redactText(one)
		if !changed {
			return body, fields
		}
		fields["input"] = marshal(out)
	case json.Unmarshal(raw, &many) == nil:
		changed := false
		for i, text := range many {
			if out, ok := s.redactText(text); ok {
				many[i], changed = out, true
			}
		}
		if !changed {
			return body, fields
		}
		fields["input"] = marshal(many)
	default:
		return body, fields
	}
	return marshal(fields), fields
}

func (s *Server) redactText(text string) (string, bool) {
	out, counts := s.redactor.Text(text)
	for kind, n := range counts {
		s.metrics.Redactions.WithLabelValues(string(kind)).Add(float64(n))
	}
	return out, counts != nil
}

// marshal encodes without escaping <, > and &, so text that wasn't redacted
// goes on as it was written.
func marshal(v any) json.RawMessage {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
	return bytes.TrimRight(buf.Bytes(), "\n")
}

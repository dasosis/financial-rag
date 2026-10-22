package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// checkStructured checks a backup's answer to a json_schema call: one choice,
// finished, and either a refusal (a well-formed answer the app handles) or
// content that is JSON matching the schema. The primary (OpenAI, strict
// structured outputs) is trusted; a backup's answer goes out only if it passes.
func checkStructured(answer []byte, schema json.RawMessage) error {
	var c struct {
		Choices []struct {
			Message struct {
				Content *string `json:"content"`
				Refusal *string `json:"refusal"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(answer, &c); err != nil {
		return fmt.Errorf("not a chat completion: %w", err)
	}
	if len(c.Choices) != 1 {
		return fmt.Errorf("%d choices", len(c.Choices))
	}
	choice := c.Choices[0]
	if choice.FinishReason != "stop" {
		return fmt.Errorf("finished %q", choice.FinishReason)
	}
	if choice.Message.Refusal != nil && *choice.Message.Refusal != "" {
		return nil
	}
	if choice.Message.Content == nil {
		return errors.New("no content")
	}
	var value any
	if err := json.Unmarshal([]byte(*choice.Message.Content), &value); err != nil {
		return fmt.Errorf("content isn't JSON: %w", err)
	}
	var root map[string]any
	if err := json.Unmarshal(schema, &root); err != nil {
		return fmt.Errorf("the request's schema isn't JSON: %w", err)
	}
	return matches(value, root, root, "$")
}

// matches checks a value against the subset of JSON Schema structured outputs
// use: type, properties, required, additionalProperties: false, items, enum,
// const, anyOf, and $ref into $defs.
func matches(value any, schema, root map[string]any, at string) error {
	if ref, ok := schema["$ref"].(string); ok {
		target, err := resolve(ref, root)
		if err != nil {
			return fmt.Errorf("%s: %w", at, err)
		}
		return matches(value, target, root, at)
	}
	if options, ok := schema["anyOf"].([]any); ok {
		for _, option := range options {
			if o, ok := option.(map[string]any); ok && matches(value, o, root, at) == nil {
				return nil
			}
		}
		return fmt.Errorf("%s: matches none of anyOf", at)
	}
	if want, ok := schema["const"]; ok && !equal(value, want) {
		return fmt.Errorf("%s: not the const", at)
	}
	if enum, ok := schema["enum"].([]any); ok && !slices.ContainsFunc(enum, func(e any) bool { return equal(value, e) }) {
		return fmt.Errorf("%s: not one of the enum", at)
	}
	if t, ok := schema["type"]; ok && !typed(value, t) {
		return fmt.Errorf("%s: not of type %v", at, t)
	}
	switch v := value.(type) {
	case map[string]any:
		props, _ := schema["properties"].(map[string]any)
		if required, ok := schema["required"].([]any); ok {
			for _, name := range required {
				if n, ok := name.(string); ok {
					if _, present := v[n]; !present {
						return fmt.Errorf("%s: %q missing", at, n)
					}
				}
			}
		}
		for name, item := range v {
			sub, known := props[name].(map[string]any)
			if !known {
				if extra, ok := schema["additionalProperties"].(bool); ok && !extra {
					return fmt.Errorf("%s: unexpected %q", at, name)
				}
				continue
			}
			if err := matches(item, sub, root, at+"."+name); err != nil {
				return err
			}
		}
	case []any:
		if items, ok := schema["items"].(map[string]any); ok {
			for i, item := range v {
				if err := matches(item, items, root, fmt.Sprintf("%s[%d]", at, i)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func resolve(ref string, root map[string]any) (map[string]any, error) {
	path, ok := strings.CutPrefix(ref, "#/")
	if !ok {
		return nil, fmt.Errorf("unsupported $ref %q", ref)
	}
	var node any = root
	for _, part := range strings.Split(path, "/") {
		m, ok := node.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("unresolvable $ref %q", ref)
		}
		node = m[part]
	}
	target, ok := node.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unresolvable $ref %q", ref)
	}
	return target, nil
}

func typed(value, t any) bool {
	if types, ok := t.([]any); ok {
		return slices.ContainsFunc(types, func(one any) bool { return typed(value, one) })
	}
	switch t {
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "number":
		_, ok := value.(float64)
		return ok
	case "integer":
		f, ok := value.(float64)
		return ok && f == float64(int64(f))
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "null":
		return value == nil
	}
	return true // a type it doesn't know: not this check's to refuse
}

func equal(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}

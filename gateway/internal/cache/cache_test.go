package cache

import (
	"encoding/json"
	"strings"
	"testing"
)

func fields(t *testing.T, body string) map[string]json.RawMessage {
	t.Helper()
	var f map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func key(t *testing.T, body string) (string, bool) {
	t.Helper()
	return Key(fields(t, body))
}

const base = `{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"What was revenue in fiscal 2024?"}]}`

func TestTheSameRequestHasTheSameKeyHoweverItIsWritten(t *testing.T) {
	a, ok := key(t, base)
	if !ok {
		t.Fatal("a temperature-0 call isn't cacheable")
	}
	same := []string{
		// fields and object keys in another order, other spacing, another spelling of 0
		`{ "messages": [ {"content": "What was revenue in fiscal 2024?", "role": "user"} ], "temperature": 0.0, "model": "gpt-4o-mini" }`,
		// how it is delivered or attributed doesn't change the answer
		`{"model":"gpt-4o-mini","temperature":0,"stream":true,"stream_options":{"include_usage":true},"user":"u","messages":[{"role":"user","content":"What was revenue in fiscal 2024?"}]}`,
		`{"model":"gpt-4o-mini","temperature":0,"n":1,"messages":[{"role":"user","content":"What was revenue in fiscal 2024?"}]}`,
	}
	for _, body := range same {
		if b, _ := key(t, body); b != a {
			t.Errorf("different key for %s", body)
		}
	}
}

func TestAnythingThatChangesTheAnswerChangesTheKey(t *testing.T) {
	a, _ := key(t, base)
	different := map[string]string{
		"another year":    strings.Replace(base, "2024", "2023", 1),
		"another model":   strings.Replace(base, "gpt-4o-mini", "gpt-4o", 1),
		"a system prompt": strings.Replace(base, `[{"role":"user"`, `[{"role":"system","content":"Be brief."},{"role":"user"`, 1),
		"tools":           strings.Replace(base, `"temperature":0`, `"temperature":0,"tools":[{"type":"function","function":{"name":"f"}}]`, 1),
		"response_format": strings.Replace(base, `"temperature":0`, `"temperature":0,"response_format":{"type":"json_object"}`, 1),
		"max_tokens":      strings.Replace(base, `"temperature":0`, `"temperature":0,"max_tokens":5`, 1),
		"seed":            strings.Replace(base, `"temperature":0`, `"temperature":0,"seed":7`, 1),
		"top_p":           strings.Replace(base, `"temperature":0`, `"temperature":0,"top_p":0.5`, 1),
	}
	for name, body := range different {
		if b, ok := key(t, body); !ok || b == a {
			t.Errorf("%s: same key (or not cacheable)", name)
		}
	}
}

func TestOnlyTemperatureZeroSingleAnswersAreCached(t *testing.T) {
	for name, body := range map[string]string{
		"no temperature (defaults to 1)": `{"model":"m","messages":[]}`,
		"temperature 0.7":                `{"model":"m","temperature":0.7,"messages":[]}`,
		"two answers":                    `{"model":"m","temperature":0,"n":2,"messages":[]}`,
	} {
		if _, ok := key(t, body); ok {
			t.Errorf("%s: cacheable", name)
		}
	}
}

func TestOnlyNormallyFinishedAnswersAreStored(t *testing.T) {
	answer := func(finish string) []byte {
		return []byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":` + finish + `}]}`)
	}
	for finish, want := range map[string]bool{
		`"stop"`: true, `"tool_calls"`: true, `"length"`: false, `"content_filter"`: false, `null`: false,
	} {
		if Finished(answer(finish)) != want {
			t.Errorf("finish %s: stored %v", finish, !want)
		}
	}
	if Finished([]byte(`{"error":{"message":"x"}}`)) {
		t.Error("an error would be stored")
	}
}

func TestAStreamIsAssembledAndReplayed(t *testing.T) {
	var a Assembler
	for _, chunk := range []string{
		`{"id":"c1","object":"chat.completion.chunk","created":5,"model":"gpt-4o-mini","system_fingerprint":"fp","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":null}`,
		`{"id":"c1","created":5,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Revenue was "},"finish_reason":null}],"usage":null}`,
		`{"id":"c1","created":5,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"$245,122 million."},"finish_reason":null}],"usage":null}`,
		`{"id":"c1","created":5,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}`,
		`{"id":"c1","created":5,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16}}`,
	} {
		a.Add([]byte(chunk))
	}
	body, ok := a.Completion()
	if !ok || !Finished(body) {
		t.Fatalf("not assembled: %s", body)
	}
	var c struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(body, &c)
	if c.ID != "c1" || c.Object != "chat.completion" || c.Choices[0].Message.Content != "Revenue was $245,122 million." ||
		c.Usage.CompletionTokens != 6 {
		t.Fatalf("assembled %s", body)
	}

	for _, usage := range []bool{false, true} {
		sse, err := Replay(body, usage)
		if err != nil {
			t.Fatal(err)
		}
		text := string(sse)
		if !strings.HasSuffix(text, "data: [DONE]\n\n") || !strings.Contains(text, `"content":"Revenue was $245,122 million."`) ||
			!strings.Contains(text, `"finish_reason":"stop"`) {
			t.Fatalf("replay %s", text)
		}
		if got := strings.Contains(text, `"completion_tokens":6`); got != usage {
			t.Fatalf("usage chunk present %v, want %v", got, usage)
		}
		// and the replay assembles back to the same answer
		var again Assembler
		for _, event := range strings.Split(strings.TrimSpace(text), "\n\n") {
			data := strings.TrimPrefix(event, "data: ")
			if data != "[DONE]" {
				again.Add([]byte(data))
			}
		}
		if rebuilt, _ := again.Completion(); usage && string(rebuilt) != string(body) {
			t.Fatalf("replay doesn't round-trip:\n%s\n%s", rebuilt, body)
		}
	}
}

func TestToolCallsAreAssembled(t *testing.T) {
	var a Assembler
	a.Add([]byte(`{"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"plan","arguments":""}}]},"finish_reason":null}]}`))
	a.Add([]byte(`{"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"route\":"}}]},"finish_reason":null}]}`))
	a.Add([]byte(`{"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"text\"}"}}]},"finish_reason":"tool_calls"}]}`))
	body, ok := a.Completion()
	if !ok || !strings.Contains(string(body), `"arguments":"{\"route\":\"text\"}"`) || !strings.Contains(string(body), `"content":null`) {
		t.Fatalf("assembled %s", body)
	}
}

func TestStreamsTheCacheCantHoldAreNotStored(t *testing.T) {
	var cut Assembler // no finish reason: the stream broke off
	cut.Add([]byte(`{"id":"c","choices":[{"index":0,"delta":{"content":"Rev"},"finish_reason":null}]}`))
	if _, ok := cut.Completion(); ok {
		t.Error("a stream without a finish was assembled")
	}
	var logprobs Assembler
	logprobs.Add([]byte(`{"id":"c","choices":[{"index":0,"delta":{"content":"a"},"logprobs":{"content":[]},"finish_reason":"stop"}]}`))
	if _, ok := logprobs.Completion(); ok {
		t.Error("a stream with logprobs was assembled")
	}
}

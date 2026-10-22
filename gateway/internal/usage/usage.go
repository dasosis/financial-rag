// Package usage is the token count a provider reports for one call.
package usage

// Usage is the "usage" object of a chat completion or embeddings response.
type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
}

// Cached is how many of the prompt tokens the provider had cached.
func (u Usage) Cached() int {
	if u.PromptTokensDetails == nil {
		return 0
	}
	return u.PromptTokensDetails.CachedTokens
}

// Known is true when the provider reported anything.
func (u Usage) Known() bool { return u.PromptTokens > 0 || u.CompletionTokens > 0 }

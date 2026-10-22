package server

import (
	"encoding/json"
	"maps"
	"net/http"
	"strings"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

func (s *Server) embeddings(c *call) {
	body, ok := s.readBody(c)
	if !ok {
		return
	}
	var fields map[string]json.RawMessage
	var model string
	if json.Unmarshal(body, &fields) != nil || json.Unmarshal(fields["model"], &model) != nil || model == "" {
		s.fail(c, http.StatusBadRequest, "invalid_request_error", "invalid_body",
			"The body must be a JSON embeddings request with a model.")
		return
	}
	// About four characters a token; the input is all there is to pay for
	reservation, ok := s.admit(c, model, usage.Usage{PromptTokens: len(body)/4 + 1})
	if !ok {
		return
	}
	// Vectors from another model can't be compared with the stored index: the
	// call fails over only to a provider serving this same model, or gets 503.
	plan := func(p *Provider) (outbound, bool) {
		served, ok := p.Model(model)
		if !ok || !SameEmbeddingModel(model, served) {
			return outbound{}, false
		}
		if served == model {
			return outbound{body: body}, true
		}
		f := maps.Clone(fields)
		f["model"], _ = json.Marshal(served)
		out, _ := json.Marshal(f)
		return outbound{body: out}, true
	}
	resp, p, _, ok := s.dispatch(c, "/embeddings", plan)
	if !ok {
		s.settle(c, reservation, 0)
		return
	}
	defer resp.Close()
	if resp.StatusCode != http.StatusOK {
		s.relayError(c, resp)
		s.settle(c, reservation, 0)
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
	copyHeaders(c.w.Header(), resp.Header, "Content-Type", "x-request-id")
	c.w.WriteHeader(http.StatusOK)
	_, _ = c.w.Write(answer)
	s.served(p, c)
	s.settle(c, reservation, s.account(model, used))
	s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint, "provider", p.Name,
		"model", model, "ms", time.Since(c.start).Milliseconds())
}

// SameEmbeddingModel is true when `served` is the model asked for, under the
// provider's name for it ("openai/text-embedding-3-small").
func SameEmbeddingModel(requested, served string) bool {
	return served == requested || strings.HasSuffix(served, "/"+requested)
}

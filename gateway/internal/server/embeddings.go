package server

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

func (s *Server) embeddings(c *call) {
	body, ok := s.readBody(c)
	if !ok {
		return
	}
	var req struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &req) != nil || req.Model == "" {
		s.fail(c, http.StatusBadRequest, "invalid_request_error", "invalid_body",
			"The body must be a JSON embeddings request with a model.")
		return
	}
	// About four characters a token; the input is all there is to pay for
	reservation, ok := s.admit(c, req.Model, usage.Usage{PromptTokens: len(body)/4 + 1})
	if !ok {
		return
	}
	resp, err := s.forward(c, "/embeddings", body, s.cfg.EmbedTimeout)
	if err != nil {
		s.settle(c, reservation, 0)
		return
	}
	defer resp.Close()
	if resp.StatusCode != http.StatusOK {
		s.relayError(c, resp)
		s.settle(c, reservation, 0)
		return
	}
	used := s.relayJSON(c, resp)
	s.settle(c, reservation, s.account(req.Model, used))
	s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint,
		"model", req.Model, "ms", time.Since(c.start).Milliseconds())
}

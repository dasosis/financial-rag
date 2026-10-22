package server

import (
	"encoding/json"
	"net/http"
	"time"
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
	resp, err := s.forward(c, "/embeddings", body, s.cfg.EmbedTimeout)
	if err != nil {
		return
	}
	defer resp.Close()
	if resp.StatusCode != http.StatusOK {
		s.relayError(c, resp)
		return
	}
	usage := s.relayJSON(c, resp)
	s.countTokens(req.Model, usage)
	s.log.Info("answered", "request_id", c.requestID, "endpoint", c.endpoint,
		"model", req.Model, "ms", time.Since(c.start).Milliseconds())
}

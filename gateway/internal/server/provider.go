package server

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/breaker"
)

// Provider is an OpenAI-compatible API the gateway forwards calls to.
type Provider struct {
	Name    string
	BaseURL string // ".../v1"
	key     string
	client  *http.Client
	// models maps the model a caller asks for to this provider's name for it.
	// nil for the primary, which serves every model under the caller's name.
	models map[string]string
	// JSONSchema: the provider follows response_format json_schema. Answers from
	// a backup are still checked against the schema before they go out.
	JSONSchema bool
	Breaker    *breaker.Breaker
}

// NewProvider returns a provider with its own connection pool. models nil:
// it serves every model as named (the primary).
func NewProvider(name, baseURL, key string, models map[string]string, b *breaker.Breaker) *Provider {
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          64,
		MaxIdleConnsPerHost:   32,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &Provider{Name: name, BaseURL: baseURL, key: key, models: models, JSONSchema: true,
		Breaker: b, client: &http.Client{Transport: transport}}
}

// Primary is true for the provider every call goes to first.
func (p *Provider) Primary() bool { return p.models == nil }

// Model is this provider's name for a requested model, if it serves it.
func (p *Provider) Model(requested string) (string, bool) {
	if p.models == nil {
		return requested, true
	}
	m, ok := p.models[requested]
	return m, ok
}

// errFirstByteTimeout means the provider sent nothing back in time.
var errFirstByteTimeout = errors.New("no response from the provider in time")

// upstreamResponse is a provider's answer, headers received. Read the body
// from `body` (it may hold the first byte already); Close releases it all.
type upstreamResponse struct {
	*http.Response
	body   io.Reader
	cancel context.CancelFunc
}

func (u *upstreamResponse) Close() {
	_ = u.Body.Close()
	u.cancel()
}

// send posts a JSON body to the provider and returns once it has started
// answering, or errFirstByteTimeout if it hasn't within `firstByte`. "Started"
// is the response headers, or for a stream (firstBodyByte) the first byte of
// the body as well: a provider can send headers at once and then stall, and
// until it sends something the call can still go to another provider. After
// that the body takes as long as it takes, bounded only by the caller's
// context: a caller that hangs up cancels the call.
func (p *Provider) send(ctx context.Context, path string, body []byte, firstByte time.Duration, firstBodyByte bool) (*upstreamResponse, error) {
	ctx, cancel := context.WithCancel(ctx)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.key)
	req.Header.Set("Content-Type", "application/json")

	var timedOut atomic.Bool
	timer := time.AfterFunc(firstByte, func() {
		timedOut.Store(true)
		cancel()
	})
	fail := func(err error) (*upstreamResponse, error) {
		timer.Stop()
		cancel()
		if timedOut.Load() {
			return nil, errFirstByteTimeout
		}
		return nil, err
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return fail(err)
	}
	reader := io.Reader(resp.Body)
	if firstBodyByte && resp.StatusCode == http.StatusOK {
		buffered := bufio.NewReaderSize(resp.Body, 64<<10)
		if _, err := buffered.Peek(1); err != nil {
			_ = resp.Body.Close()
			return fail(err)
		}
		reader = buffered
	}
	timer.Stop()
	if timedOut.Load() { // the timer fired just as the answer arrived
		_ = resp.Body.Close()
		cancel()
		return nil, errFirstByteTimeout
	}
	return &upstreamResponse{Response: resp, body: reader, cancel: cancel}, nil
}

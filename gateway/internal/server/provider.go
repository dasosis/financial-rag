package server

import (
	"bytes"
	"context"
	"errors"
	"net"
	"net/http"
	"sync/atomic"
	"time"
)

// Provider is an OpenAI-compatible API the gateway forwards calls to.
type Provider struct {
	Name    string
	BaseURL string // ".../v1"
	key     string
	client  *http.Client
}

// NewProvider returns a provider with its own connection pool.
func NewProvider(name, baseURL, key string) *Provider {
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
	return &Provider{Name: name, BaseURL: baseURL, key: key, client: &http.Client{Transport: transport}}
}

// errFirstByteTimeout means the provider sent nothing back in time.
var errFirstByteTimeout = errors.New("no response from the provider in time")

// upstreamResponse is a provider's answer, headers received. Close releases it.
type upstreamResponse struct {
	*http.Response
	cancel context.CancelFunc
}

func (u *upstreamResponse) Close() {
	_ = u.Body.Close()
	u.cancel()
}

// send posts a JSON body to the provider and returns once its response headers
// arrive, or errFirstByteTimeout if they don't within `firstByte`. After that
// the body may take as long as it takes (a stream is still being written),
// bounded only by the caller's context: a caller that hangs up cancels the call.
func (p *Provider) send(ctx context.Context, path string, body []byte, firstByte time.Duration) (*upstreamResponse, error) {
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
	resp, err := p.client.Do(req)
	timer.Stop()
	if err != nil {
		cancel()
		if timedOut.Load() {
			return nil, errFirstByteTimeout
		}
		return nil, err
	}
	if timedOut.Load() { // the timer fired as the headers arrived
		_ = resp.Body.Close()
		cancel()
		return nil, errFirstByteTimeout
	}
	return &upstreamResponse{Response: resp, cancel: cancel}, nil
}

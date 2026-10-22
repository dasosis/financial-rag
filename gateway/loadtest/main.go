// Command loadtest drives the gateway (or the fake upstream directly, to
// compare) with concurrent chat calls and reports what happened. Run it
// against the FAKE upstream only (cmd/fakeupstream): it makes thousands of
// calls, and against OpenAI each would cost money.
//
//	loadtest -url http://llm-gateway:8080/v1 -key $LLM_GATEWAY_KEY -n 5000 -c 20
//	loadtest -url http://fake:9100/v1 -n 5000 -c 20              # no gateway: the baseline
//	loadtest -scenario cache -n 2000 -c 20                        # one question, asked again and again
//	loadtest -scenario limit -duration 20s -c 50 -rate 600 -burst 60
//
// Scenarios:
//
//	overhead  every call a different question (no cache hits), timing each one
//	cache     the same temperature-0 question every time: after the first, all hits
//	limit     many consumers as fast as they can for -duration; reports how many
//	          calls the global limit (-rate a minute, -burst) let through against
//	          the most it allows: burst + rate x elapsed
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	url := flag.String("url", "http://127.0.0.1:8080/v1", "base URL (the gateway, or the fake upstream)")
	key := flag.String("key", os.Getenv("LLM_GATEWAY_KEY"), "the gateway key")
	scenario := flag.String("scenario", "overhead", "overhead, cache or limit")
	n := flag.Int("n", 2000, "calls (overhead, cache)")
	c := flag.Int("c", 20, "concurrent clients")
	duration := flag.Duration("duration", 20*time.Second, "how long to run (limit)")
	rate := flag.Float64("rate", 0, "the global limit being tested, calls a minute (limit)")
	burst := flag.Float64("burst", 0, "the global burst being tested (limit)")
	stream := flag.Bool("stream", false, "streamed calls")
	flag.Parse()

	// Hex, not digits: a long run of digits can pass for a card number, and the
	// gateway would redact it (making questions from different runs the same)
	run := strconv.FormatInt(time.Now().UnixNano(), 16)
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: *c * 2}, Timeout: time.Minute}
	// The limit scenario must measure the limiter alone: temperature 1 is never
	// cached (a cache hit doesn't count against any limit)
	temperature := 0.0
	if *scenario == "limit" {
		temperature = 1
	}
	call := func(consumer string, i int, repeat bool) (int, time.Duration, time.Duration, error) {
		// A run id in every question: temperature-0 calls from an earlier run
		// would otherwise be answered from the cache
		question := fmt.Sprintf("Load test %s, question %d: what was revenue?", run, i)
		if repeat {
			question = "Load test: what was revenue in fiscal 2024?"
		}
		body, _ := json.Marshal(map[string]any{
			"model": "gpt-4o-mini", "temperature": temperature, "stream": *stream,
			"messages": []any{map[string]any{"role": "user", "content": question}},
		})
		req, _ := http.NewRequest(http.MethodPost, *url+"/chat/completions", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+*key)
		req.Header.Set("X-Consumer-Id", consumer)
		start := time.Now()
		resp, err := client.Do(req)
		if err != nil {
			return 0, 0, 0, err
		}
		defer resp.Body.Close()
		first := make([]byte, 1)
		_, _ = io.ReadFull(resp.Body, first)
		ttfb := time.Since(start)
		_, _ = io.Copy(io.Discard, resp.Body)
		return resp.StatusCode, time.Since(start), ttfb, nil
	}

	switch *scenario {
	case "overhead", "cache":
		repeat := *scenario == "cache"
		if repeat {
			call("user:loadtest-0", 0, true) // the miss that fills the cache
		}
		var mu sync.Mutex
		var latencies, ttfbs []time.Duration
		codes := map[int]int{}
		var next atomic.Int64
		var wg sync.WaitGroup
		start := time.Now()
		for w := range *c {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i := int(next.Add(1))
					if i > *n {
						return
					}
					code, total, ttfb, err := call(fmt.Sprintf("user:loadtest-%d", w), i, repeat)
					mu.Lock()
					if err != nil {
						codes[-1]++
					} else {
						codes[code]++
						latencies = append(latencies, total)
						ttfbs = append(ttfbs, ttfb)
					}
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		elapsed := time.Since(start)
		fmt.Printf("scenario=%s url=%s calls=%d concurrency=%d stream=%v elapsed=%.1fs throughput=%.0f/s codes=%v\n",
			*scenario, *url, *n, *c, *stream, elapsed.Seconds(), float64(*n)/elapsed.Seconds(), codes)
		fmt.Printf("latency  p50=%s p95=%s p99=%s max=%s\n", q(latencies, 0.5), q(latencies, 0.95), q(latencies, 0.99), q(latencies, 1))
		fmt.Printf("ttfb     p50=%s p95=%s p99=%s\n", q(ttfbs, 0.5), q(ttfbs, 0.95), q(ttfbs, 0.99))

	case "limit":
		var ok, limited, other atomic.Int64
		deadline := time.Now().Add(*duration)
		var wg sync.WaitGroup
		start := time.Now()
		for w := range *c {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; time.Now().Before(deadline); i++ {
					code, _, _, err := call(fmt.Sprintf("user:limit-%d", w), w*1_000_000+i, false)
					switch {
					case err != nil:
						other.Add(1)
					case code == http.StatusOK:
						ok.Add(1)
					case code == http.StatusTooManyRequests:
						limited.Add(1)
					default:
						other.Add(1)
					}
				}
			}()
		}
		wg.Wait()
		elapsed := time.Since(start).Seconds()
		bound := *burst + *rate/60*elapsed
		fmt.Printf("scenario=limit url=%s concurrency=%d elapsed=%.1fs admitted=%d limited=%d other=%d\n",
			*url, *c, elapsed, ok.Load(), limited.Load(), other.Load())
		if bound > 0 {
			fmt.Printf("global limit: at most %.1f (burst %.0f + %.0f/min x %.1fs); admitted %d = %.2f%% of the limit\n",
				bound, *burst, *rate, elapsed, ok.Load(), 100*float64(ok.Load())/bound)
		}
	default:
		fmt.Fprintln(os.Stderr, "unknown -scenario")
		os.Exit(2)
	}
}

// q is the q-th quantile of the durations (1 = the maximum).
func q(d []time.Duration, quantile float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := slices.Clone(d)
	slices.Sort(s)
	i := int(quantile * float64(len(s)-1))
	return s[i].Round(10 * time.Microsecond)
}

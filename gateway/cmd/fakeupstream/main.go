// Command fakeupstream serves a fake OpenAI API on a local port, for the
// Python integration test and the load test: nothing it does costs money.
//
//	fakeupstream -addr 127.0.0.1:9100 -latency 50ms -chunk-delay 5ms
//
// Point the gateway at it with OPENAI_BASE_URL=http://127.0.0.1:9100/v1.
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/fakeupstream"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:9100", "address to listen on")
	latency := flag.Duration("latency", 0, "delay before each response starts")
	chunkDelay := flag.Duration("chunk-delay", 0, "delay between streamed chunks")
	model := flag.String("model", "", "model name to report (default: the one asked for)")
	flag.Parse()

	fake := fakeupstream.New(fakeupstream.Options{Latency: *latency, ChunkDelay: *chunkDelay, Model: *model})
	srv := &http.Server{Addr: *addr, Handler: fake, ReadHeaderTimeout: 10 * time.Second}
	log.Printf("fake OpenAI API on http://%s/v1", *addr)
	log.Fatal(srv.ListenAndServe())
}

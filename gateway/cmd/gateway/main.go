// Command gateway is the LLM gateway: an OpenAI-compatible API between the app
// and the model providers. See gateway/README.md and the main README's
// "LLM gateway" section.
//
//	gateway                serve (configured from the environment)
//	gateway -healthcheck   exit 0 if the local server answers /healthz (for Docker)
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/dasosis/financial-rag/gateway/internal/config"
	"github.com/dasosis/financial-rag/gateway/internal/metrics"
	"github.com/dasosis/financial-rag/gateway/internal/server"
)

func main() {
	healthcheck := flag.Bool("healthcheck", false, "check the local server's /healthz and exit")
	flag.Parse()
	if *healthcheck {
		os.Exit(checkHealth())
	}

	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cfg, err := config.FromEnv()
	if err != nil {
		log.Error("refusing to start", "error", err.Error())
		os.Exit(2)
	}
	if err := run(cfg, log); err != nil {
		log.Error("stopped", "error", err.Error())
		os.Exit(1)
	}
}

func run(cfg config.Config, log *slog.Logger) error {
	m := metrics.New()
	srv := server.New(cfg, m, log)

	api := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// No ReadTimeout or WriteTimeout: answers stream for minutes. The body
		// read has a deadline of its own (server.readBody).
		IdleTimeout: 120 * time.Second,
	}
	metricsMux := http.NewServeMux()
	metricsMux.Handle("GET /metrics", m.Handler())
	internal := &http.Server{Addr: cfg.MetricsAddr, Handler: metricsMux, ReadHeaderTimeout: 5 * time.Second}

	errs := make(chan error, 2)
	go func() { errs <- serve(api) }()
	go func() { errs <- serve(internal) }()
	log.Info("listening", "addr", cfg.Addr, "metrics", cfg.MetricsAddr, "openai", cfg.OpenAIBaseURL)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	select {
	case err := <-errs:
		return err
	case sig := <-stop:
		log.Info("shutting down", "signal", sig.String())
	}
	// Leave the load balancer's rotation first, then let calls in flight finish.
	srv.SetReady(false)
	time.Sleep(2 * time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = internal.Shutdown(ctx)
	return api.Shutdown(ctx)
}

func serve(s *http.Server) error {
	if err := s.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func checkHealth() int {
	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	if strings.HasPrefix(addr, ":") {
		addr = "127.0.0.1" + addr
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://" + addr + "/healthz")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}

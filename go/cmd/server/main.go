// Command server runs the Go gRPC inference service with dynamic batching.
//
// Beyond the happy path it wires the controls a Kubernetes deployment needs:
// validation with precise status codes, client-deadline propagation, admission
// control, model-version pinning for canaries, pipelined streaming, keepalive
// tuned for L4 load balancers, Prometheus metrics on a side port, and a
// shutdown that fails readiness before it stops accepting work.
package main

import (
	"context"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	pb "github.com/aabhimittal/grpc-kubernetes/go/gen/inferencev1"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/batch"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/metrics"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/validate"
)

type server struct {
	pb.UnimplementedInferenceServiceServer
	registry          *inference.Registry
	batcher           *batch.Batcher
	metrics           *metrics.Registry
	streamConcurrency int
	ready             atomic.Bool
}

func (s *server) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	start := time.Now()
	s.metrics.Inc("inference_requests_total", 1)
	res, err := s.predict(ctx, req)
	if err != nil {
		return nil, err
	}
	s.metrics.Observe("inference_latency_ms", float64(time.Since(start).Microseconds())/1000.0)
	s.metrics.Inc("inference_responses_total", 1)
	return s.toResponse(req.RequestId, res, start), nil
}

// PredictStream is pipelined: requests are dispatched concurrently and
// responses are emitted as they complete, so one slow item cannot stall the
// stream and a single client can actually fill a batch. Responses may be
// reordered — correlate on request_id, which is what it is for.
func (s *server) PredictStream(stream pb.InferenceService_PredictStreamServer) error {
	ctx := stream.Context()
	sem := make(chan struct{}, s.streamConcurrency)
	out := make(chan *pb.PredictResponse, s.streamConcurrency)
	errCh := make(chan error, 1)

	failOnce := func(err error) {
		select {
		case errCh <- err:
		default: // first error wins; the rest are consequences of it
		}
	}

	go func() {
		var wg sync.WaitGroup
		for {
			req, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break // clean client CloseSend
			}
			if err != nil {
				failOnce(err)
				break
			}
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				failOnce(ctx.Err())
				goto drain
			}
			wg.Add(1)
			go func(req *pb.PredictRequest) {
				defer wg.Done()
				defer func() { <-sem }()
				start := time.Now()
				s.metrics.Inc("inference_stream_requests_total", 1)
				res, err := s.predict(ctx, req)
				if err != nil {
					failOnce(err)
					return
				}
				s.metrics.Observe("inference_latency_ms",
					float64(time.Since(start).Microseconds())/1000.0)
				select {
				case out <- s.toResponse(req.RequestId, res, start):
				case <-ctx.Done():
				}
			}(req)
		}
	drain:
		wg.Wait()
		close(out)
	}()

	for resp := range out {
		if err := stream.Send(resp); err != nil {
			go func() { //nolint:errcheck // unblock producers, result discarded
				for range out {
				}
			}()
			return err
		}
		s.metrics.Inc("inference_stream_responses_total", 1)
	}
	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

func (s *server) HealthCheck(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	st := pb.HealthResponse_NOT_SERVING
	if s.ready.Load() {
		st = pb.HealthResponse_SERVING
	}
	return &pb.HealthResponse{Status: st, ModelVersion: s.registry.Default}, nil
}

// predict validates, routes and submits one request, translating every failure
// mode into the gRPC code a client can act on: retry elsewhere, back off, or
// fix the request.
func (s *server) predict(ctx context.Context, req *pb.PredictRequest) (batch.Result, error) {
	model, err := s.registry.Get(req.ModelVersion)
	if err != nil {
		s.metrics.Inc("inference_errors_total", 1)
		return batch.Result{}, status.Errorf(codes.NotFound,
			"unknown model_version %q; loaded: %v", req.ModelVersion, s.registry.Versions())
	}
	if err := validate.Features(req.Features, model.Dim()); err != nil {
		s.metrics.Inc("inference_invalid_total", 1)
		return batch.Result{}, status.Error(codes.InvalidArgument, err.Error())
	}

	res, err := s.batcher.Submit(ctx, req.Features, req.ModelVersion)
	switch {
	case err == nil:
		return res, nil
	case errors.Is(err, batch.ErrQueueFull):
		s.metrics.Inc("inference_rejected_total", 1)
		return batch.Result{}, status.Error(codes.ResourceExhausted, err.Error())
	case errors.Is(err, batch.ErrExpired), errors.Is(err, context.DeadlineExceeded):
		s.metrics.Inc("inference_expired_total", 1)
		return batch.Result{}, status.Error(codes.DeadlineExceeded, err.Error())
	case errors.Is(err, batch.ErrShuttingDown):
		return batch.Result{}, status.Error(codes.Unavailable, err.Error())
	case errors.Is(err, context.Canceled):
		return batch.Result{}, status.Error(codes.Canceled, err.Error())
	default:
		s.metrics.Inc("inference_errors_total", 1)
		return batch.Result{}, status.Error(codes.Internal, err.Error())
	}
}

func (s *server) toResponse(id string, res batch.Result, start time.Time) *pb.PredictResponse {
	label := int32(0)
	best := float32(-1)
	for i, v := range res.Scores {
		if v > best {
			best, label = v, int32(i)
		}
	}
	return &pb.PredictResponse{
		RequestId:       id,
		Scores:          res.Scores,
		Label:           label,
		ModelVersion:    res.ModelVersion,
		ServerLatencyMs: float64(time.Since(start).Microseconds()) / 1000.0,
		BatchSize:       int32(res.BatchSize),
	}
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// grpcOptions: server options matter as much as the handlers once this runs
// behind a Service with long-lived HTTP/2 connections.
func grpcOptions() []grpc.ServerOption {
	return []grpc.ServerOption{
		grpc.KeepaliveParams(keepalive.ServerParameters{
			// Force clients to re-resolve periodically so a scale-out actually
			// rebalances long-lived connections (docs/LLD.md §5).
			MaxConnectionAge:      time.Duration(envInt("MAX_CONN_AGE_MS", 300_000)) * time.Millisecond,
			MaxConnectionAgeGrace: 10 * time.Second,
			Time:                  time.Duration(envInt("KEEPALIVE_MS", 30_000)) * time.Millisecond,
			Timeout:               10 * time.Second,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			// Tolerate keepalives from idle clients instead of GOAWAY-ing them.
			MinTime:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.MaxRecvMsgSize(envInt("MAX_MSG_BYTES", 4*1024*1024)),
		grpc.MaxSendMsgSize(envInt("MAX_MSG_BYTES", 4*1024*1024)),
		grpc.MaxConcurrentStreams(uint32(envInt("MAX_CONCURRENT_STREAMS", 1000))),
	}
}

func main() {
	port := getenv("PORT", "50051")
	registry, err := inference.RegistryFromSpec(
		getenv("MODEL_VERSIONS", getenv("MODEL_VERSION", "v1")),
		os.Getenv("MODEL_VERSION"),
	)
	if err != nil {
		log.Fatalf("model registry: %v", err)
	}
	mreg := metrics.NewRegistry()
	batcher, err := batch.New(registry, batch.Config{
		MaxBatch: envInt("MAX_BATCH", 32),
		MaxDelay: time.Duration(envInt("MAX_DELAY_MS", 5)) * time.Millisecond,
		MinDelay: time.Duration(envInt("MIN_DELAY_US", 200)) * time.Microsecond,
		MaxQueue: envInt("MAX_QUEUE", 1024),
		Adaptive: getenv("ADAPTIVE_BATCHING", "1") != "0",
		Metrics:  mreg,
	})
	if err != nil {
		log.Fatalf("batcher config: %v", err)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()
	go batcher.Run(ctx)

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	grpcServer := grpc.NewServer(grpcOptions()...)
	srv := &server{
		registry:          registry,
		batcher:           batcher,
		metrics:           mreg,
		streamConcurrency: envInt("STREAM_CONCURRENCY", 32),
	}
	pb.RegisterInferenceServiceServer(grpcServer, srv)

	// Standard grpc.health.v1 service — this is what Kubernetes native gRPC
	// probes and grpc_health_probe query. Report NOT_SERVING until warm.
	hs := health.NewServer()
	healthpb.RegisterHealthServer(grpcServer, hs)
	hs.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)

	metricsSrv := &http.Server{
		Addr:              ":" + strconv.Itoa(envInt("METRICS_PORT", 9090)),
		Handler:           mreg.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		if err := metricsSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("metrics server: %v", err)
		}
	}()

	registry.WarmupAll()
	srv.ready.Store(true)
	hs.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)

	go func() {
		<-ctx.Done()
		log.Println("shutting down")
		// Fail readiness first so kube-proxy pulls this pod from the Service
		// before in-flight work is disturbed (docs/LLD.md §6).
		hs.Shutdown()
		srv.ready.Store(false)
		if d := envInt("PRESTOP_SLEEP_MS", 0); d > 0 {
			time.Sleep(time.Duration(d) * time.Millisecond)
		}
		stopped := make(chan struct{})
		go func() { defer close(stopped); grpcServer.GracefulStop() }()
		select {
		case <-stopped:
		case <-time.After(time.Duration(envInt("GRACE_MS", 10_000)) * time.Millisecond):
			log.Println("grace period elapsed; forcing stop")
			grpcServer.Stop()
		}
		shutCtx, cancelShut := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelShut()
		_ = metricsSrv.Shutdown(shutCtx)
	}()

	log.Printf("go inference server listening on :%s (metrics :%d, versions %v)",
		port, envInt("METRICS_PORT", 9090), registry.Versions())
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

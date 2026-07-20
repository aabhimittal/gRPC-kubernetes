// Command server runs the Go gRPC inference service with dynamic batching.
package main

import (
	"context"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/aabhimittal/grpc-kubernetes/go/gen/inferencev1"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/batch"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
)

type server struct {
	pb.UnimplementedInferenceServiceServer
	model   *inference.Model
	batcher *batch.Batcher
	ready   atomic.Bool
}

func (s *server) Predict(ctx context.Context, req *pb.PredictRequest) (*pb.PredictResponse, error) {
	start := time.Now()
	res, err := s.batcher.Submit(ctx, req.Features)
	if err != nil {
		return nil, err
	}
	return s.toResponse(req.RequestId, res, start), nil
}

func (s *server) PredictStream(stream pb.InferenceService_PredictStreamServer) error {
	ctx := stream.Context()
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil // clean client CloseSend
		}
		if err != nil {
			return err
		}
		start := time.Now()
		res, err := s.batcher.Submit(ctx, req.Features)
		if err != nil {
			return err
		}
		if err := stream.Send(s.toResponse(req.RequestId, res, start)); err != nil {
			return err
		}
	}
}

func (s *server) HealthCheck(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	st := pb.HealthResponse_NOT_SERVING
	if s.ready.Load() {
		st = pb.HealthResponse_SERVING
	}
	return &pb.HealthResponse{Status: st, ModelVersion: s.model.Version}, nil
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
		ModelVersion:    s.model.Version,
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

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}
	model := inference.New(getenv("MODEL_VERSION", "v1"))
	batcher := batch.New(model, envInt("MAX_BATCH", 32),
		time.Duration(envInt("MAX_DELAY_MS", 5))*time.Millisecond)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()
	go batcher.Run(ctx)

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	srv := &server{model: model, batcher: batcher}
	pb.RegisterInferenceServiceServer(grpcServer, srv)

	// Standard grpc.health.v1 service — this is what Kubernetes native gRPC
	// probes and grpc_health_probe query. Report NOT_SERVING until warm.
	hs := health.NewServer()
	healthpb.RegisterHealthServer(grpcServer, hs)
	hs.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)

	model.Warmup()
	srv.ready.Store(true)
	hs.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)

	go func() {
		<-ctx.Done()
		log.Println("shutting down")
		hs.Shutdown() // flip health to NOT_SERVING so probes fail fast
		grpcServer.GracefulStop()
	}()

	log.Printf("go inference server listening on :%s", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

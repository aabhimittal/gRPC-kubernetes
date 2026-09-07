package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/aabhimittal/grpc-kubernetes/go/gen/inferencev1"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/batch"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/metrics"
)

// End-to-end tests over a real gRPC connection (bufconn): they assert the
// status codes a client actually has to branch on, which is where a serving
// tier most often disappoints its callers.

type harness struct {
	client pb.InferenceServiceClient
	srv    *server
	stop   func()
}

func newHarness(t *testing.T, spec string, cfg batch.Config, runLoop bool) *harness {
	t.Helper()
	registry, err := inference.RegistryFromSpec(spec, "")
	if err != nil {
		t.Fatal(err)
	}
	mreg := metrics.NewRegistry()
	batcher, err := batch.New(registry, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	if runLoop {
		go batcher.Run(ctx)
	}

	srv := &server{registry: registry, batcher: batcher, metrics: mreg, streamConcurrency: 16}
	srv.ready.Store(true)

	lis := bufconn.Listen(1024 * 1024)
	grpcServer := grpc.NewServer()
	pb.RegisterInferenceServiceServer(grpcServer, srv)
	go func() { _ = grpcServer.Serve(lis) }()

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	return &harness{
		client: pb.NewInferenceServiceClient(conn),
		srv:    srv,
		stop: func() {
			_ = conn.Close()
			grpcServer.Stop()
			cancel()
		},
	}
}

func feats(v float32) []float32 {
	f := make([]float32, inference.FeatureDim)
	for i := range f {
		f[i] = v
	}
	return f
}

func codeOf(err error) codes.Code { return status.Code(err) }

func TestPredictHappyPath(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{MaxDelay: 5 * time.Millisecond}, true)
	defer h.stop()

	resp, err := h.client.Predict(context.Background(),
		&pb.PredictRequest{RequestId: "r1", Features: feats(0.5)})
	if err != nil {
		t.Fatal(err)
	}
	if resp.RequestId != "r1" || resp.ModelVersion != "v1" || resp.BatchSize < 1 {
		t.Fatalf("unexpected response %+v", resp)
	}
	var sum float64
	for _, s := range resp.Scores {
		sum += float64(s)
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("scores sum to %v", sum)
	}
	if int(resp.Label) >= len(resp.Scores) {
		t.Fatalf("label %d out of range", resp.Label)
	}
}

func TestInvalidRequestsGetInvalidArgument(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{MaxDelay: 5 * time.Millisecond}, true)
	defer h.stop()

	nan := feats(0)
	nan[3] = float32(math.NaN())
	cases := map[string][]float32{
		"empty":     {},
		"truncated": make([]float32, inference.FeatureDim-1),
		"too long":  make([]float32, inference.FeatureDim+1),
		"nan":       nan,
	}
	for name, f := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := h.client.Predict(context.Background(), &pb.PredictRequest{Features: f})
			if codeOf(err) != codes.InvalidArgument {
				t.Fatalf("got %v (%v), want InvalidArgument", codeOf(err), err)
			}
		})
	}
}

func TestUnknownModelVersionGetsNotFound(t *testing.T) {
	h := newHarness(t, "v1,v2", batch.Config{MaxDelay: 5 * time.Millisecond}, true)
	defer h.stop()

	_, err := h.client.Predict(context.Background(),
		&pb.PredictRequest{Features: feats(0.1), ModelVersion: "v99"})
	if codeOf(err) != codes.NotFound {
		t.Fatalf("got %v, want NotFound", codeOf(err))
	}
	if msg := status.Convert(err).Message(); msg == "" {
		t.Fatal("error should name the versions that are loaded")
	}
}

func TestVersionPinRoutesToThatModel(t *testing.T) {
	h := newHarness(t, "v1:1,v2:2", batch.Config{MaxDelay: 5 * time.Millisecond}, true)
	defer h.stop()

	a, err := h.client.Predict(context.Background(),
		&pb.PredictRequest{Features: feats(0.5), ModelVersion: "v1"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := h.client.Predict(context.Background(),
		&pb.PredictRequest{Features: feats(0.5), ModelVersion: "v2"})
	if err != nil {
		t.Fatal(err)
	}
	if a.ModelVersion != "v1" || b.ModelVersion != "v2" {
		t.Fatalf("pins ignored: %q %q", a.ModelVersion, b.ModelVersion)
	}
	if a.Scores[0] == b.Scores[0] {
		t.Fatal("canary returned the incumbent's answer")
	}
}

func TestExpiredDeadlineGetsDeadlineExceeded(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{MaxDelay: 5 * time.Millisecond}, true)
	defer h.stop()

	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	time.Sleep(time.Millisecond)
	_, err := h.client.Predict(ctx, &pb.PredictRequest{Features: feats(0.1)})
	if codeOf(err) != codes.DeadlineExceeded {
		t.Fatalf("got %v, want DeadlineExceeded", codeOf(err))
	}
}

func TestOverloadShedsWithResourceExhausted(t *testing.T) {
	// Batcher loop deliberately not running: the queue fills and admission
	// control must shed rather than let latency grow without bound.
	h := newHarness(t, "v1", batch.Config{MaxQueue: 1, MaxDelay: time.Second}, false)
	defer h.stop()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	go func() { _, _ = h.client.Predict(ctx, &pb.PredictRequest{Features: feats(0.1)}) }()

	var got codes.Code
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		shortCtx, cancelShort := context.WithTimeout(context.Background(), 500*time.Millisecond)
		_, err := h.client.Predict(shortCtx, &pb.PredictRequest{Features: feats(0.1)})
		cancelShort()
		if got = codeOf(err); got == codes.ResourceExhausted {
			return
		}
	}
	t.Fatalf("never shed load; last code %v", got)
}

func TestHealthCheckReflectsReadiness(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{}, true)
	defer h.stop()

	resp, err := h.client.HealthCheck(context.Background(), &pb.HealthRequest{})
	if err != nil || resp.Status != pb.HealthResponse_SERVING {
		t.Fatalf("got %v, %v; want SERVING", resp, err)
	}
	h.srv.ready.Store(false)
	resp, err = h.client.HealthCheck(context.Background(), &pb.HealthRequest{})
	if err != nil || resp.Status != pb.HealthResponse_NOT_SERVING {
		t.Fatalf("got %v, %v; want NOT_SERVING", resp, err)
	}
}

func TestStreamReturnsEveryRequestAndIsPipelined(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{MaxBatch: 32, MaxDelay: 20 * time.Millisecond}, true)
	defer h.stop()

	stream, err := h.client.PredictStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	const n = 24
	for i := 0; i < n; i++ {
		if err := stream.Send(&pb.PredictRequest{
			RequestId: fmt.Sprintf("r%d", i), Features: feats(float32(i) / 10),
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatal(err)
	}

	seen := map[string]bool{}
	maxBatch := int32(0)
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if seen[resp.RequestId] {
			t.Fatalf("duplicate response for %s", resp.RequestId)
		}
		seen[resp.RequestId] = true
		if resp.BatchSize > maxBatch {
			maxBatch = resp.BatchSize
		}
	}
	if len(seen) != n {
		t.Fatalf("got %d responses, want %d", len(seen), n)
	}
	// Pipelining is the point: a serialized stream would batch one at a time.
	if maxBatch < 2 {
		t.Fatalf("stream never batched (max batch_size %d) — is it pipelined?", maxBatch)
	}
}

func TestStreamSurfacesPerRequestErrors(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{MaxDelay: 5 * time.Millisecond}, true)
	defer h.stop()

	stream, err := h.client.PredictStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Send(&pb.PredictRequest{RequestId: "bad", Features: nil}); err != nil {
		t.Fatal(err)
	}
	_ = stream.CloseSend()
	for {
		_, err := stream.Recv()
		if err == nil {
			continue
		}
		if codeOf(err) != codes.InvalidArgument {
			t.Fatalf("got %v, want InvalidArgument", codeOf(err))
		}
		return
	}
}

func TestConcurrentUnaryCallsAreBatched(t *testing.T) {
	h := newHarness(t, "v1", batch.Config{MaxBatch: 64, MaxDelay: 25 * time.Millisecond}, true)
	defer h.stop()

	const n = 50
	var wg sync.WaitGroup
	var mu sync.Mutex
	maxBatch := int32(0)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			resp, err := h.client.Predict(context.Background(),
				&pb.PredictRequest{RequestId: fmt.Sprintf("c%d", i), Features: feats(0.25)})
			if err != nil {
				t.Errorf("call %d: %v", i, err)
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if resp.BatchSize > maxBatch {
				maxBatch = resp.BatchSize
			}
		}(i)
	}
	wg.Wait()
	if maxBatch < 2 {
		t.Fatalf("no coalescing under %d concurrent calls (max batch %d)", n, maxBatch)
	}
}

// Command client is a demo / load generator for the inference service.
//
//	go run ./cmd/client -target localhost:50051 -mode unary
//	go run ./cmd/client -target localhost:50051 -mode stream -n 100
//	go run ./cmd/client -target localhost:50051 -mode load -n 500 -concurrency 50
//
// The load mode is deliberately failure-tolerant: it tallies gRPC status codes
// instead of dying on the first shed request, because the interesting runs are
// exactly the ones where the server starts returning RESOURCE_EXHAUSTED or
// DEADLINE_EXCEEDED.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	pb "github.com/aabhimittal/grpc-kubernetes/go/gen/inferencev1"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/metrics"
)

// serviceConfig: round_robin spreads streams across pods behind a headless
// Service, and the retry policy turns the server's shed/drain signals into
// transparent retries instead of user-visible errors. Only codes the server
// returns *before* touching the model are retried, so retries are safe.
const serviceConfig = `{
  "loadBalancingConfig": [{"round_robin":{}}],
  "methodConfig": [{
    "name": [{"service": "inference.v1.InferenceService"}],
    "retryPolicy": {
      "maxAttempts": 4,
      "initialBackoff": "0.05s",
      "maxBackoff": "1s",
      "backoffMultiplier": 2,
      "retryableStatusCodes": ["UNAVAILABLE", "RESOURCE_EXHAUSTED"]
    }
  }]
}`

func req(i int, version string) *pb.PredictRequest {
	f := make([]float32, inference.FeatureDim)
	for j := range f {
		f[j] = rand.Float32()
	}
	return &pb.PredictRequest{
		RequestId:    fmt.Sprintf("r%d", i),
		Features:     f,
		ModelVersion: version,
	}
}

type tally struct {
	mu     sync.Mutex
	counts map[codes.Code]int
}

func newTally() *tally { return &tally{counts: map[codes.Code]int{}} }

func (t *tally) add(err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.counts[status.Code(err)]++
}

func (t *tally) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	keys := make([]string, 0, len(t.counts))
	for c, n := range t.counts {
		keys = append(keys, fmt.Sprintf("%s=%d", c, n))
	}
	sort.Strings(keys)
	return fmt.Sprintf("%v", keys)
}

func main() {
	target := flag.String("target", "localhost:50051", "server address")
	mode := flag.String("mode", "unary", "unary|stream|load")
	n := flag.Int("n", 100, "number of requests")
	concurrency := flag.Int("concurrency", 50, "load mode concurrency")
	version := flag.String("version", "", "model_version pin (empty = server default)")
	timeout := flag.Duration("timeout", 5*time.Second, "per-RPC deadline")
	flag.Parse()

	conn, err := grpc.NewClient(*target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(serviceConfig),
		// Keep the HTTP/2 connection alive across idle gaps so the next
		// request does not pay a fresh handshake.
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time: 30 * time.Second, Timeout: 10 * time.Second, PermitWithoutStream: true,
		}),
	)
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	stub := pb.NewInferenceServiceClient(conn)

	switch *mode {
	case "unary":
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		defer cancel()
		resp, err := stub.Predict(ctx, req(0, *version))
		if err != nil {
			log.Fatalf("%s: %v", status.Code(err), err)
		}
		fmt.Printf("label=%d version=%s batch_size=%d latency=%.2fms scores=%v\n",
			resp.Label, resp.ModelVersion, resp.BatchSize, resp.ServerLatencyMs, resp.Scores)

	case "stream":
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		defer cancel()
		stream, err := stub.PredictStream(ctx)
		if err != nil {
			log.Fatal(err)
		}
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			seen, maxBatch := 0, int32(0)
			for {
				resp, err := stream.Recv()
				if err == io.EOF {
					break
				}
				if err != nil {
					log.Printf("stream ended: %s: %v", status.Code(err), err)
					break
				}
				seen++
				if resp.BatchSize > maxBatch {
					maxBatch = resp.BatchSize
				}
			}
			fmt.Printf("streamed %d predictions (max server batch_size=%d)\n", seen, maxBatch)
		}()
		for i := 0; i < *n; i++ {
			if err := stream.Send(req(i, *version)); err != nil {
				log.Printf("send: %v", err)
				break
			}
		}
		_ = stream.CloseSend()
		wg.Wait()

	case "load":
		var sum, ok int64
		lat := metrics.NewHistogram(nil, *n+1)
		codeTally := newTally()
		sem := make(chan struct{}, *concurrency)
		var wg sync.WaitGroup
		start := time.Now()
		for i := 0; i < *n; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(i int) {
				defer wg.Done()
				defer func() { <-sem }()
				ctx, cancel := context.WithTimeout(context.Background(), *timeout)
				defer cancel()
				t0 := time.Now()
				resp, err := stub.Predict(ctx, req(i, *version))
				codeTally.add(err)
				if err != nil {
					return // tallied, not fatal: shedding is a valid outcome
				}
				lat.Observe(float64(time.Since(t0).Microseconds()) / 1000.0)
				atomic.AddInt64(&sum, int64(resp.BatchSize))
				atomic.AddInt64(&ok, 1)
			}(i)
		}
		wg.Wait()
		elapsed := time.Since(start)
		avgBatch := 0.0
		if ok > 0 {
			avgBatch = float64(sum) / float64(ok)
		}
		fmt.Printf("sent %d @ concurrency %d in %v (%.0f rps)\n",
			*n, *concurrency, elapsed.Round(time.Millisecond),
			float64(*n)/elapsed.Seconds())
		fmt.Printf("  ok=%d codes=%s avg server batch_size=%.1f\n", ok, codeTally, avgBatch)
		fmt.Printf("  client latency p50=%.1fms p95=%.1fms p99=%.1fms\n",
			lat.Percentile(0.5), lat.Percentile(0.95), lat.Percentile(0.99))

	default:
		log.Fatalf("unknown mode %q", *mode)
	}
}

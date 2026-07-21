// Command client is a demo / load generator for the inference service.
//
//	go run ./cmd/client -target localhost:50051 -mode unary
//	go run ./cmd/client -target localhost:50051 -mode stream -n 100
//	go run ./cmd/client -target localhost:50051 -mode load  -n 500 -concurrency 50
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/aabhimittal/grpc-kubernetes/go/gen/inferencev1"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
)

func req(i int) *pb.PredictRequest {
	f := make([]float32, inference.FeatureDim)
	for j := range f {
		f[j] = rand.Float32()
	}
	return &pb.PredictRequest{RequestId: fmt.Sprintf("r%d", i), Features: f}
}

func main() {
	target := flag.String("target", "localhost:50051", "server address")
	mode := flag.String("mode", "unary", "unary|stream|load")
	n := flag.Int("n", 100, "number of requests")
	concurrency := flag.Int("concurrency", 50, "load mode concurrency")
	flag.Parse()

	// round_robin spreads streams across pods behind a headless Service.
	conn, err := grpc.NewClient(*target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(`{"loadBalancingConfig":[{"round_robin":{}}]}`),
	)
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	stub := pb.NewInferenceServiceClient(conn)
	ctx := context.Background()

	switch *mode {
	case "unary":
		resp, err := stub.Predict(ctx, req(0))
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("label=%d batch_size=%d latency=%.2fms scores=%v\n",
			resp.Label, resp.BatchSize, resp.ServerLatencyMs, resp.Scores)

	case "stream":
		stream, err := stub.PredictStream(ctx)
		if err != nil {
			log.Fatal(err)
		}
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			seen := 0
			for {
				if _, err := stream.Recv(); err == io.EOF {
					break
				} else if err != nil {
					log.Fatal(err)
				}
				seen++
			}
			fmt.Printf("streamed %d predictions\n", seen)
		}()
		for i := 0; i < *n; i++ {
			if err := stream.Send(req(i)); err != nil {
				log.Fatal(err)
			}
		}
		stream.CloseSend()
		wg.Wait()

	case "load":
		var sum, cnt int64
		sem := make(chan struct{}, *concurrency)
		var wg sync.WaitGroup
		start := time.Now()
		for i := 0; i < *n; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(i int) {
				defer wg.Done()
				defer func() { <-sem }()
				resp, err := stub.Predict(ctx, req(i))
				if err != nil {
					log.Fatal(err)
				}
				atomic.AddInt64(&sum, int64(resp.BatchSize))
				atomic.AddInt64(&cnt, 1)
			}(i)
		}
		wg.Wait()
		fmt.Printf("sent %d requests @ concurrency %d in %v; avg server batch_size=%.1f\n",
			*n, *concurrency, time.Since(start), float64(sum)/float64(cnt))
	}
}

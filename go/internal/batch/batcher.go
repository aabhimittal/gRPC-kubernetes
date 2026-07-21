// Package batch implements the dynamic batcher described in docs/LLD.md §2.
package batch

import (
	"context"
	"time"

	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
)

// Result is delivered back to the waiting RPC handler.
type Result struct {
	Scores    []float32
	BatchSize int
}

type item struct {
	features []float32
	respCh   chan Result
}

// Batcher coalesces concurrent requests into single model calls.
type Batcher struct {
	model    *inference.Model
	maxBatch int
	maxDelay time.Duration
	in       chan item
}

func New(model *inference.Model, maxBatch int, maxDelay time.Duration) *Batcher {
	return &Batcher{
		model:    model,
		maxBatch: maxBatch,
		maxDelay: maxDelay,
		in:       make(chan item, 1024),
	}
}

// Submit enqueues one request and blocks until its result is ready.
func (b *Batcher) Submit(ctx context.Context, features []float32) (Result, error) {
	respCh := make(chan Result, 1)
	select {
	case b.in <- item{features: features, respCh: respCh}:
	case <-ctx.Done():
		return Result{}, ctx.Err()
	}
	select {
	case r := <-respCh:
		return r, nil
	case <-ctx.Done():
		return Result{}, ctx.Err()
	}
}

// Run is the single consumer loop. Cancel ctx to drain and stop.
func (b *Batcher) Run(ctx context.Context) {
	for {
		var first item
		select {
		case <-ctx.Done():
			return
		case first = <-b.in:
		}

		batch := []item{first}
		timer := time.NewTimer(b.maxDelay)
	fill:
		for len(batch) < b.maxBatch {
			select {
			case it := <-b.in:
				batch = append(batch, it)
			case <-timer.C:
				break fill
			case <-ctx.Done():
				break fill
			}
		}
		timer.Stop()

		features := make([][]float32, len(batch))
		for i, it := range batch {
			features[i] = it.features
		}
		outputs := b.model.PredictBatch(features)
		n := len(batch)
		for i, it := range batch {
			it.respCh <- Result{Scores: outputs[i], BatchSize: n}
		}
	}
}

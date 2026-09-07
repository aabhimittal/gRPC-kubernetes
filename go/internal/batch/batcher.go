// Package batch implements the dynamic batcher described in docs/LLD.md §2.
//
// Beyond plain coalescing it carries the controls a production inference tier
// needs once traffic stops being polite:
//
//   - Admission control — a bounded queue. Past its depth, work is rejected
//     immediately (ErrQueueFull -> RESOURCE_EXHAUSTED) rather than building an
//     unbounded latency tail that outlives every client's deadline.
//   - Deadline-aware batching — queued work whose client deadline already
//     passed is dropped before the model call, so a latency spike does not
//     spend accelerator time on answers nobody is waiting for.
//   - Adaptive window — the wait shrinks toward MinDelay while batches keep
//     filling and grows back toward MaxDelay when traffic thins, so a quiet
//     service does not pay a fixed per-request tax.
//   - Version-aware grouping — requests pinned to different model versions in
//     one window become separate model calls, so canary traffic neither blocks
//     nor blends into the incumbent's batch.
//   - Draining shutdown — queued work is served or failed fast with a
//     retryable error instead of hanging until the client's own deadline.
package batch

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/metrics"
)

var (
	// ErrQueueFull means admission control shed this request. Retryable.
	ErrQueueFull = errors.New("inference queue full")
	// ErrExpired means the client deadline passed before compute started.
	ErrExpired = errors.New("deadline exceeded while queued")
	// ErrShuttingDown means the server is draining. Retryable elsewhere.
	ErrShuttingDown = errors.New("server is shutting down")
)

// Result is delivered back to the waiting RPC handler.
type Result struct {
	Scores       []float32
	BatchSize    int
	ModelVersion string
	QueueWait    time.Duration
}

type response struct {
	result Result
	err    error
}

type item struct {
	features []float32
	version  string
	deadline time.Time // zero value = no deadline
	enqueued time.Time
	respCh   chan response
}

func (it item) expired(now time.Time) bool {
	return !it.deadline.IsZero() && !now.Before(it.deadline)
}

// Config carries the tunables; zero values fall back to sane defaults so a
// caller can specify only what it cares about.
type Config struct {
	MaxBatch int
	MaxDelay time.Duration
	MinDelay time.Duration
	MaxQueue int
	Adaptive bool
	Metrics  *metrics.Registry
}

func (c Config) withDefaults() (Config, error) {
	if c.MaxBatch == 0 {
		c.MaxBatch = 32
	}
	if c.MaxDelay == 0 {
		c.MaxDelay = 5 * time.Millisecond
	}
	if c.MinDelay == 0 {
		c.MinDelay = 200 * time.Microsecond
	}
	if c.MaxQueue == 0 {
		c.MaxQueue = 1024
	}
	switch {
	case c.MaxBatch < 1:
		return c, fmt.Errorf("MaxBatch must be >= 1, got %d", c.MaxBatch)
	case c.MaxQueue < 1:
		return c, fmt.Errorf("MaxQueue must be >= 1, got %d", c.MaxQueue)
	case c.MinDelay > c.MaxDelay:
		return c, fmt.Errorf("MinDelay %v exceeds MaxDelay %v", c.MinDelay, c.MaxDelay)
	}
	return c, nil
}

// Batcher coalesces concurrent requests into single model calls.
type Batcher struct {
	registry *inference.Registry
	cfg      Config
	in       chan item
	done     chan struct{} // closed once Run has returned and drained
	delay    atomic.Int64  // current adaptive window, nanoseconds
}

// New validates the configuration up front: a batcher misconfigured at startup
// should fail the pod, not silently serve with surprising latency.
func New(registry *inference.Registry, cfg Config) (*Batcher, error) {
	cfg, err := cfg.withDefaults()
	if err != nil {
		return nil, err
	}
	b := &Batcher{
		registry: registry,
		cfg:      cfg,
		in:       make(chan item, cfg.MaxQueue),
		done:     make(chan struct{}),
	}
	b.delay.Store(int64(cfg.MaxDelay))
	return b, nil
}

// QueueDepth is what the HPA and the /metrics gauge care about.
func (b *Batcher) QueueDepth() int { return len(b.in) }

// CurrentDelay reports the adaptive window in effect right now.
func (b *Batcher) CurrentDelay() time.Duration { return time.Duration(b.delay.Load()) }

// Submit enqueues one request and blocks until its result is ready, the client
// context ends, or the batcher rejects it.
func (b *Batcher) Submit(ctx context.Context, features []float32, version string) (Result, error) {
	if _, err := b.registry.Get(version); err != nil {
		return Result{}, err // unknown pin: never take a queue slot
	}
	now := time.Now()
	deadline, hasDeadline := ctx.Deadline()
	if hasDeadline && !now.Before(deadline) {
		b.cfg.Metrics.Inc("inference_requests_expired_total", 1)
		return Result{}, ErrExpired
	}

	select {
	case <-b.done:
		return Result{}, ErrShuttingDown
	default:
	}

	it := item{
		features: features,
		version:  version,
		enqueued: now,
		respCh:   make(chan response, 1), // buffered: the loop never blocks
	}
	if hasDeadline {
		it.deadline = deadline
	}

	select {
	case b.in <- it:
	default:
		b.cfg.Metrics.Inc("inference_requests_rejected_total", 1)
		return Result{}, fmt.Errorf("%w (%d deep)", ErrQueueFull, b.cfg.MaxQueue)
	}
	b.cfg.Metrics.SetGauge("inference_queue_depth", float64(len(b.in)))

	select {
	case r := <-it.respCh:
		return r.result, r.err
	case <-ctx.Done():
		return Result{}, ctx.Err()
	case <-b.done:
		return Result{}, ErrShuttingDown
	}
}

// Run is the single consumer loop. Cancel ctx to drain and stop: work already
// queued is served, then anything still waiting is failed fast.
func (b *Batcher) Run(ctx context.Context) {
	defer close(b.done)
	defer b.rejectPending()

	for {
		var first item
		select {
		case first = <-b.in:
		case <-ctx.Done():
			// Graceful drain: serve what is already queued, then stop.
			select {
			case first = <-b.in:
			default:
				return
			}
		}
		b.process(b.fill(ctx, first))
	}
}

func (b *Batcher) fill(ctx context.Context, first item) []item {
	batch := make([]item, 0, b.cfg.MaxBatch)
	batch = append(batch, first)
	timer := time.NewTimer(b.CurrentDelay())
	defer timer.Stop()

fill:
	for len(batch) < b.cfg.MaxBatch {
		select {
		case it := <-b.in:
			batch = append(batch, it)
		case <-timer.C:
			break fill
		case <-ctx.Done():
			// Shutting down: take whatever is already queued, do not wait.
			for len(batch) < b.cfg.MaxBatch {
				select {
				case it := <-b.in:
					batch = append(batch, it)
				default:
					break fill
				}
			}
			break fill
		}
	}
	b.adapt(len(batch))
	return batch
}

// adapt halves the window while batches saturate (arrival rate is high enough
// that waiting buys nothing) and eases it back when batches arrive alone.
func (b *Batcher) adapt(filled int) {
	if !b.cfg.Adaptive {
		return
	}
	cur := b.CurrentDelay()
	switch {
	case filled >= b.cfg.MaxBatch:
		if next := cur / 2; next > b.cfg.MinDelay {
			b.delay.Store(int64(next))
		} else {
			b.delay.Store(int64(b.cfg.MinDelay))
		}
	case filled == 1:
		if next := cur * 2; next < b.cfg.MaxDelay {
			b.delay.Store(int64(next))
		} else {
			b.delay.Store(int64(b.cfg.MaxDelay))
		}
	}
}

func (b *Batcher) process(batch []item) {
	now := time.Now()
	live := map[string][]item{}
	for _, it := range batch {
		if it.expired(now) {
			b.cfg.Metrics.Inc("inference_requests_expired_total", 1)
			it.respCh <- response{err: ErrExpired}
			continue
		}
		version := it.version
		if version == "" {
			version = b.registry.Default
		}
		live[version] = append(live[version], it)
	}
	b.cfg.Metrics.SetGauge("inference_queue_depth", float64(len(b.in)))

	for version, items := range live {
		model, err := b.registry.Get(version)
		if err != nil { // registry mutated under us; fail only this group
			b.cfg.Metrics.Inc("inference_model_errors_total", 1)
			for _, it := range items {
				it.respCh <- response{err: err}
			}
			continue
		}
		features := make([][]float32, len(items))
		for i, it := range items {
			features[i] = it.features
		}
		outputs := model.PredictBatch(features)
		n := len(items)
		b.cfg.Metrics.Inc("inference_batches_total", 1)
		b.cfg.Metrics.Inc("inference_batched_requests_total", float64(n))
		b.cfg.Metrics.Observe("inference_batch_size", float64(n))
		for i, it := range items {
			it.respCh <- response{result: Result{
				Scores:       outputs[i],
				BatchSize:    n,
				ModelVersion: model.Version,
				QueueWait:    now.Sub(it.enqueued),
			}}
		}
	}
}

func (b *Batcher) rejectPending() {
	for {
		select {
		case it := <-b.in:
			it.respCh <- response{err: ErrShuttingDown}
		default:
			return
		}
	}
}

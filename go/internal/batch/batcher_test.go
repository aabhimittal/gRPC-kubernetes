package batch

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/aabhimittal/grpc-kubernetes/go/internal/inference"
	"github.com/aabhimittal/grpc-kubernetes/go/internal/metrics"
)

func features() []float32 { return make([]float32, inference.FeatureDim) }

func registry(t *testing.T, spec, def string) *inference.Registry {
	t.Helper()
	reg, err := inference.RegistryFromSpec(spec, def)
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

// newRunning returns a started batcher and a stop func that waits for the loop
// to finish draining, so tests never leak goroutines.
func newRunning(t *testing.T, cfg Config) (*Batcher, func()) {
	t.Helper()
	b, err := New(registry(t, "v1", "v1"), cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); b.Run(ctx) }()
	return b, func() {
		cancel()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("batcher loop did not stop")
		}
	}
}

func TestConcurrentRequestsCoalesceIntoOneBatch(t *testing.T) {
	b, stop := newRunning(t, Config{MaxBatch: 32, MaxDelay: 25 * time.Millisecond})
	defer stop()

	const n = 8
	sizes := make([]int, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			res, err := b.Submit(context.Background(), features(), "")
			if err != nil {
				t.Errorf("submit: %v", err)
				return
			}
			sizes[i] = res.BatchSize
		}(i)
	}
	wg.Wait()
	for i, s := range sizes {
		if s != n {
			t.Fatalf("request %d saw batch_size %d, want %d", i, s, n)
		}
	}
}

func TestBatchNeverExceedsMaxBatch(t *testing.T) {
	b, stop := newRunning(t, Config{MaxBatch: 4, MaxDelay: 25 * time.Millisecond})
	defer stop()

	var wg sync.WaitGroup
	var mu sync.Mutex
	var maxSeen, count int
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := b.Submit(context.Background(), features(), "")
			if err != nil {
				t.Errorf("submit: %v", err)
				return
			}
			mu.Lock()
			defer mu.Unlock()
			count++
			if res.BatchSize > maxSeen {
				maxSeen = res.BatchSize
			}
		}()
	}
	wg.Wait()
	if count != 20 {
		t.Fatalf("served %d of 20", count)
	}
	if maxSeen > 4 {
		t.Fatalf("batch grew to %d, max is 4", maxSeen)
	}
}

func TestQueueFullShedsLoadInsteadOfQueueingForever(t *testing.T) {
	// No Run loop: nothing drains, so the queue fills and admission control
	// must reject rather than block.
	b, err := New(registry(t, "v1", "v1"), Config{MaxBatch: 8, MaxQueue: 2})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for i := 0; i < 2; i++ {
		go func() { _, _ = b.Submit(ctx, features(), "") }()
	}
	deadline := time.Now().Add(time.Second)
	for b.QueueDepth() < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if _, err := b.Submit(context.Background(), features(), ""); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("got %v, want ErrQueueFull", err)
	}
}

func TestDeadlineAlreadyPassedIsRefusedWithoutQueueing(t *testing.T) {
	b, stop := newRunning(t, Config{})
	defer stop()

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Millisecond))
	defer cancel()
	if _, err := b.Submit(ctx, features(), ""); !errors.Is(err, ErrExpired) {
		t.Fatalf("got %v, want ErrExpired", err)
	}
	if b.QueueDepth() != 0 {
		t.Fatal("expired request consumed a queue slot")
	}
}

func TestExpiredWhileQueuedIsDroppedBeforeCompute(t *testing.T) {
	// Loop starts only after the deadline has lapsed, mimicking a request that
	// waited out its budget behind a backlog.
	b, err := New(registry(t, "v1", "v1"), Config{MaxBatch: 8, MaxDelay: 5 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx, cancelReq := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancelReq()
	errCh := make(chan error, 1)
	go func() {
		_, err := b.Submit(reqCtx, features(), "")
		errCh <- err
	}()
	time.Sleep(80 * time.Millisecond)

	runCtx, cancelRun := context.WithCancel(context.Background())
	defer cancelRun()
	go b.Run(runCtx)

	select {
	case err := <-errCh:
		// Either the batcher dropped it, or the client's own context fired
		// first — both mean no model time was spent on dead work.
		if !errors.Is(err, ErrExpired) && !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("got %v, want ErrExpired or DeadlineExceeded", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expired request hung")
	}
}

func TestClientCancellationUnblocksTheCaller(t *testing.T) {
	b, err := New(registry(t, "v1", "v1"), Config{MaxQueue: 4})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		_, err := b.Submit(ctx, features(), "")
		errCh <- err
	}()
	time.Sleep(10 * time.Millisecond)
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("got %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("caller stayed blocked after cancellation")
	}
}

func TestUnknownVersionRefusedBeforeTakingAQueueSlot(t *testing.T) {
	b, err := New(registry(t, "v1,v2", "v1"), Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Submit(context.Background(), features(), "v9"); !errors.Is(err, inference.ErrUnknownVersion) {
		t.Fatalf("got %v, want ErrUnknownVersion", err)
	}
	if b.QueueDepth() != 0 {
		t.Fatal("rejected request consumed a queue slot")
	}
}

func TestMixedVersionsSplitIntoPerVersionBatches(t *testing.T) {
	b, err := New(registry(t, "v1:1,v2:2", "v1"), Config{MaxBatch: 32, MaxDelay: 30 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go b.Run(ctx)

	type outcome struct {
		version string
		size    int
	}
	results := make(chan outcome, 5)
	var wg sync.WaitGroup
	submit := func(version string) {
		defer wg.Done()
		res, err := b.Submit(context.Background(), features(), version)
		if err != nil {
			t.Errorf("submit %s: %v", version, err)
			return
		}
		results <- outcome{res.ModelVersion, res.BatchSize}
	}
	for _, v := range []string{"v1", "v1", "v1", "v2", "v2"} {
		wg.Add(1)
		go submit(v)
	}
	wg.Wait()
	close(results)

	sizes := map[string][]int{}
	for r := range results {
		sizes[r.version] = append(sizes[r.version], r.size)
	}
	if len(sizes["v1"]) != 3 || len(sizes["v2"]) != 2 {
		t.Fatalf("routing wrong: %v", sizes)
	}
	for _, s := range sizes["v1"] {
		if s > 3 {
			t.Fatalf("v1 batch %d includes canary traffic", s)
		}
	}
	for _, s := range sizes["v2"] {
		if s > 2 {
			t.Fatalf("v2 batch %d includes incumbent traffic", s)
		}
	}
}

func TestAdaptiveWindowShrinksUnderLoadAndRecovers(t *testing.T) {
	b, stop := newRunning(t, Config{
		MaxBatch: 2, MaxDelay: 40 * time.Millisecond,
		MinDelay: time.Millisecond, Adaptive: true,
	})
	defer stop()

	start := b.CurrentDelay()
	for i := 0; i < 4; i++ { // saturate: every window fills
		var wg sync.WaitGroup
		for j := 0; j < 2; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, err := b.Submit(context.Background(), features(), ""); err != nil {
					t.Errorf("submit: %v", err)
				}
			}()
		}
		wg.Wait()
	}
	low := b.CurrentDelay()
	if low >= start {
		t.Fatalf("window did not shrink under load: %v -> %v", start, low)
	}
	for i := 0; i < 3; i++ { // idle: batches of one
		if _, err := b.Submit(context.Background(), features(), ""); err != nil {
			t.Fatal(err)
		}
	}
	if b.CurrentDelay() <= low {
		t.Fatalf("window did not recover when idle: %v", b.CurrentDelay())
	}
}

func TestAdaptiveWindowStaysWithinBounds(t *testing.T) {
	b, stop := newRunning(t, Config{
		MaxBatch: 2, MaxDelay: 20 * time.Millisecond,
		MinDelay: 2 * time.Millisecond, Adaptive: true,
	})
	defer stop()

	for i := 0; i < 30; i++ {
		var wg sync.WaitGroup
		for j := 0; j < 2; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _ = b.Submit(context.Background(), features(), "")
			}()
		}
		wg.Wait()
	}
	if got := b.CurrentDelay(); got < 2*time.Millisecond {
		t.Fatalf("window %v fell below MinDelay", got)
	}
	for i := 0; i < 30; i++ {
		if _, err := b.Submit(context.Background(), features(), ""); err != nil {
			t.Fatal(err)
		}
	}
	if got := b.CurrentDelay(); got > 20*time.Millisecond {
		t.Fatalf("window %v exceeded MaxDelay", got)
	}
}

func TestShutdownDrainsQueuedWorkInsteadOfHangingClients(t *testing.T) {
	b, err := New(registry(t, "v1", "v1"), Config{MaxBatch: 8, MaxDelay: 5 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	loopDone := make(chan struct{})
	go func() { defer close(loopDone); b.Run(ctx) }()

	if _, err := b.Submit(context.Background(), features(), ""); err != nil {
		t.Fatal(err)
	}
	cancel()
	<-loopDone

	// Post-shutdown submissions fail fast and retryably; they never hang.
	done := make(chan error, 1)
	go func() {
		_, err := b.Submit(context.Background(), features(), "")
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, ErrShuttingDown) {
			t.Fatalf("got %v, want ErrShuttingDown", err)
		}
	case <-time.After(time.Second):
		t.Fatal("submit hung after shutdown")
	}
}

func TestMetricsRecordBatchesAndRejections(t *testing.T) {
	m := metrics.NewRegistry()
	b, err := New(registry(t, "v1", "v1"), Config{MaxBatch: 8, MaxDelay: 20 * time.Millisecond, Metrics: m})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go b.Run(ctx)

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := b.Submit(context.Background(), features(), ""); err != nil {
				t.Errorf("submit: %v", err)
			}
		}()
	}
	wg.Wait()

	if got := m.Counter("inference_batched_requests_total"); got != 4 {
		t.Fatalf("batched requests = %v, want 4", got)
	}
	if m.Counter("inference_batches_total") < 1 {
		t.Fatal("no batches counted")
	}
}

func TestInvalidConfigurationIsRejectedAtConstruction(t *testing.T) {
	reg := registry(t, "v1", "v1")
	for name, cfg := range map[string]Config{
		"negative batch": {MaxBatch: -1},
		"negative queue": {MaxQueue: -1},
		"min above max":  {MinDelay: 50 * time.Millisecond, MaxDelay: 5 * time.Millisecond},
	} {
		if _, err := New(reg, cfg); err == nil {
			t.Fatalf("%s: accepted invalid config", name)
		}
	}
}

func TestZeroConfigGetsWorkingDefaults(t *testing.T) {
	b, stop := newRunning(t, Config{})
	defer stop()
	res, err := b.Submit(context.Background(), features(), "")
	if err != nil {
		t.Fatal(err)
	}
	if res.BatchSize != 1 || res.ModelVersion != "v1" {
		t.Fatalf("unexpected result %+v", res)
	}
}

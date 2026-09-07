package metrics

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestEmptyHistogramPercentileIsZero(t *testing.T) {
	if got := NewHistogram(nil, 0).Percentile(0.99); got != 0 {
		t.Fatalf("got %v, want 0", got)
	}
}

func TestPercentilesAreNearestRank(t *testing.T) {
	h := NewHistogram(nil, 4096)
	for i := 1; i <= 100; i++ {
		h.Observe(float64(i))
	}
	for _, tc := range []struct{ q, want float64 }{
		{0, 1}, {0.5, 50}, {0.95, 95}, {0.99, 99}, {1, 100},
	} {
		if got := h.Percentile(tc.q); got != tc.want {
			t.Fatalf("p%v = %v, want %v", tc.q, got, tc.want)
		}
	}
}

func TestPercentileRejectsOutOfRange(t *testing.T) {
	h := NewHistogram(nil, 8)
	h.Observe(1)
	for _, q := range []float64{-0.1, 1.1, math.NaN()} {
		if got := h.Percentile(q); !math.IsNaN(got) {
			t.Fatalf("q=%v returned %v, want NaN", q, got)
		}
	}
}

func TestReservoirIsBoundedButCountsStayExact(t *testing.T) {
	h := NewHistogram(nil, 8)
	for i := 0; i < 1000; i++ {
		h.Observe(float64(i))
	}
	if len(h.ring) != 8 {
		t.Fatalf("reservoir grew to %d", len(h.ring))
	}
	if got := h.Percentile(1); got != 999 {
		t.Fatalf("newest sample lost: p100 = %v", got)
	}
	if h.Count() != 1000 {
		t.Fatalf("count = %d, want 1000", h.Count())
	}
}

func TestBucketsAreCumulativeAndIncludeOverflow(t *testing.T) {
	r := NewRegistry()
	r.Observe("latency_ms", 0.1)
	r.Observe("latency_ms", 10000)
	text := r.Render()
	for _, want := range []string{
		`latency_ms_bucket{le="0.5"} 1`,
		fmt.Sprintf(`latency_ms_bucket{le="%g"} 1`, DefaultBucketsMs[len(DefaultBucketsMs)-1]),
		`latency_ms_bucket{le="+Inf"} 2`,
		"latency_ms_count 2",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("render missing %q:\n%s", want, text)
		}
	}
}

func TestBoundaryValueLandsInItsOwnBucket(t *testing.T) {
	h := NewHistogram(nil, 16)
	h.Observe(5) // le is inclusive
	if h.counts[3] != 1 {
		t.Fatalf("boundary sample landed in %v", h.counts)
	}
}

func TestCountersAddAndGaugesReplace(t *testing.T) {
	r := NewRegistry()
	r.Inc("requests_total", 1)
	r.Inc("requests_total", 4)
	r.SetGauge("queue_depth", 7)
	r.SetGauge("queue_depth", 3)
	text := r.Render()
	if !strings.Contains(text, "requests_total 5") || !strings.Contains(text, "queue_depth 3") {
		t.Fatalf("bad render:\n%s", text)
	}
	if r.Counter("never_touched") != 0 {
		t.Fatal("unknown counter should read zero")
	}
}

// A nil registry is the "metrics disabled" case; call sites must stay clean.
func TestNilRegistryIsSafe(t *testing.T) {
	var r *Registry
	r.Inc("x", 1)
	r.SetGauge("y", 1)
	r.Observe("z", 1)
	if r.Render() != "" || r.Counter("x") != 0 || r.Gauge("y") != 0 {
		t.Fatal("nil registry misbehaved")
	}
}

func TestConcurrentUpdatesAreRaceFree(t *testing.T) {
	r := NewRegistry()
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				r.Inc("hits", 1)
				r.Observe("lat", float64(j))
				r.SetGauge("depth", float64(j))
			}
		}()
	}
	wg.Wait()
	if got := r.Counter("hits"); got != 3200 {
		t.Fatalf("counter = %v, want 3200 (lost updates)", got)
	}
	if got := r.Histogram("lat").Count(); got != 3200 {
		t.Fatalf("histogram count = %d, want 3200", got)
	}
}

func TestHandlerServesScrapesAnd404s(t *testing.T) {
	r := NewRegistry()
	r.Inc("inference_requests_total", 3)
	srv := httptest.NewServer(r.Handler())
	defer srv.Close()

	get := func(path string) (int, string) {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(body)
	}
	if code, body := get("/metrics"); code != 200 || !strings.Contains(body, "inference_requests_total 3") {
		t.Fatalf("metrics: %d %s", code, body)
	}
	if code, _ := get("/healthz"); code != 200 {
		t.Fatalf("healthz: %d", code)
	}
	if code, _ := get("/nope"); code != 404 {
		t.Fatalf("unknown path: %d", code)
	}
}

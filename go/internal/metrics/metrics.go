// Package metrics is a dependency-free counter/gauge/histogram registry that
// renders Prometheus text exposition. Pulling a full client library into an
// inference image costs build time and attack surface for four metric types.
package metrics

import (
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
)

// DefaultBucketsMs spans sub-millisecond to multi-second, the useful range for
// online inference latency.
var DefaultBucketsMs = []float64{0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500}

// Histogram keeps exact cumulative counts plus a bounded ring of recent samples
// for percentiles, so a pod running for weeks does not grow without bound.
type Histogram struct {
	mu      sync.Mutex
	buckets []float64
	counts  []uint64 // len(buckets)+1; last cell is +Inf
	sum     float64
	n       uint64
	ring    []float64
	cap     int
	pos     int
}

func NewHistogram(buckets []float64, reservoir int) *Histogram {
	if len(buckets) == 0 {
		buckets = DefaultBucketsMs
	}
	if reservoir <= 0 {
		reservoir = 4096
	}
	return &Histogram{buckets: buckets, counts: make([]uint64, len(buckets)+1), cap: reservoir}
}

func (h *Histogram) Observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.sum += v
	h.n++
	idx := len(h.buckets)
	for i, b := range h.buckets {
		if v <= b {
			idx = i
			break
		}
	}
	h.counts[idx]++
	if len(h.ring) < h.cap {
		h.ring = append(h.ring, v)
		return
	}
	h.ring[h.pos] = v
	h.pos = (h.pos + 1) % h.cap
}

// Percentile is nearest-rank over the reservoir; q in [0,1]. An empty
// histogram reports 0 rather than erroring — dashboards scrape cold pods too.
func (h *Histogram) Percentile(q float64) float64 {
	if q < 0 || q > 1 || math.IsNaN(q) {
		return math.NaN()
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.ring) == 0 {
		return 0
	}
	ordered := append([]float64(nil), h.ring...)
	sort.Float64s(ordered)
	rank := int(math.Ceil(q * float64(len(ordered))))
	if rank < 1 {
		rank = 1
	}
	return ordered[rank-1]
}

func (h *Histogram) Count() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.n
}

// Registry is the process-wide metric store.
type Registry struct {
	mu       sync.Mutex
	counters map[string]float64
	gauges   map[string]float64
	hists    map[string]*Histogram
}

func NewRegistry() *Registry {
	return &Registry{
		counters: map[string]float64{},
		gauges:   map[string]float64{},
		hists:    map[string]*Histogram{},
	}
}

// Inc is nil-safe so call sites need no guard when metrics are disabled.
func (r *Registry) Inc(name string, v float64) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.counters[name] += v
}

func (r *Registry) Counter(name string) float64 {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.counters[name]
}

func (r *Registry) SetGauge(name string, v float64) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.gauges[name] = v
}

func (r *Registry) Gauge(name string) float64 {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.gauges[name]
}

func (r *Registry) Histogram(name string) *Histogram {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	h, ok := r.hists[name]
	if !ok {
		h = NewHistogram(DefaultBucketsMs, 4096)
		r.hists[name] = h
	}
	return h
}

func (r *Registry) Observe(name string, v float64) {
	if r == nil {
		return
	}
	r.Histogram(name).Observe(v)
}

// Render writes Prometheus text exposition format (version 0.0.4).
func (r *Registry) Render() string {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	counters := copyFloats(r.counters)
	gauges := copyFloats(r.gauges)
	names := make([]string, 0, len(r.hists))
	hists := make(map[string]*Histogram, len(r.hists))
	for k, v := range r.hists {
		names = append(names, k)
		hists[k] = v
	}
	r.mu.Unlock()

	var b strings.Builder
	for _, name := range sortedKeys(counters) {
		fmt.Fprintf(&b, "# TYPE %s counter\n%s %g\n", name, name, counters[name])
	}
	for _, name := range sortedKeys(gauges) {
		fmt.Fprintf(&b, "# TYPE %s gauge\n%s %g\n", name, name, gauges[name])
	}
	sort.Strings(names)
	for _, name := range names {
		h := hists[name]
		h.mu.Lock()
		fmt.Fprintf(&b, "# TYPE %s histogram\n", name)
		var cum uint64
		for i, bound := range h.buckets {
			cum += h.counts[i]
			fmt.Fprintf(&b, "%s_bucket{le=\"%g\"} %d\n", name, bound, cum)
		}
		cum += h.counts[len(h.counts)-1]
		fmt.Fprintf(&b, "%s_bucket{le=\"+Inf\"} %d\n%s_sum %g\n%s_count %d\n",
			name, cum, name, h.sum, name, h.n)
		h.mu.Unlock()
	}
	return b.String()
}

// Handler serves GET /metrics and GET /healthz. Mount it on a side port so a
// scrape is never queued behind saturated inference traffic.
func (r *Registry) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = w.Write([]byte(r.Render()))
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	return mux
}

func copyFloats(in map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func sortedKeys(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Package inference holds the model stand-in. A real backend (ONNX Runtime,
// Triton, a cgo call into libtorch) implements the same PredictBatch signature.
package inference

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

const (
	FeatureDim = 16
	NumClasses = 4
	// DefaultSeed must match python/model.py so both backends agree.
	DefaultSeed = int64(1234567)
)

// ErrUnknownVersion is returned for a model_version pin that is not loaded.
// The server maps it to gRPC NOT_FOUND.
var ErrUnknownVersion = errors.New("unknown model version")

// Model mirrors python/model.py: softmax(W·x) with seeded weights so both
// implementations produce comparable outputs.
type Model struct {
	Version string
	w       [NumClasses][FeatureDim]float64
}

// Dim is the feature width this model accepts; validation rejects anything else.
func (m *Model) Dim() int { return FeatureDim }

// New builds a model with the default weights.
func New(version string) *Model { return NewWithSeed(version, DefaultSeed) }

// NewWithSeed builds a model with distinct weights, so a canary version
// actually behaves differently from the incumbent.
func NewWithSeed(version string, seed int64) *Model {
	m := &Model{Version: version}
	for c := 0; c < NumClasses; c++ {
		for f := 0; f < FeatureDim; f++ {
			seed = (seed*1103515245 + 12345) & 0x7FFFFFFF
			m.w[c][f] = float64(seed)/float64(0x7FFFFFFF)*2.0 - 1.0
		}
	}
	return m
}

// PredictBatch maps (n, FeatureDim) -> (n, NumClasses) softmax scores.
func (m *Model) PredictBatch(batch [][]float32) [][]float32 {
	out := make([][]float32, len(batch))
	for i, feats := range batch {
		var logits [NumClasses]float64
		for c := 0; c < NumClasses; c++ {
			for f := 0; f < FeatureDim; f++ {
				if f < len(feats) {
					logits[c] += m.w[c][f] * float64(feats[f])
				}
			}
		}
		max := logits[0]
		for _, v := range logits {
			if v > max {
				max = v
			}
		}
		var sum float64
		var exps [NumClasses]float64
		for c := 0; c < NumClasses; c++ {
			exps[c] = math.Exp(logits[c] - max)
			sum += exps[c]
		}
		scores := make([]float32, NumClasses)
		for c := 0; c < NumClasses; c++ {
			scores[c] = float32(exps[c] / sum)
		}
		out[i] = scores
	}
	return out
}

func (m *Model) Warmup() {
	m.PredictBatch([][]float32{make([]float32, FeatureDim)})
}

// Registry holds several model versions at once so a rollout is a routing
// change rather than a redeploy. Canary and shadow traffic both need two
// versions resident in the same process; pinning `model_version` on the
// request selects one, an empty pin follows Default.
type Registry struct {
	models  map[string]*Model
	Default string
}

// NewRegistry fails rather than silently serving the wrong weights if the
// requested default is not among the loaded versions.
func NewRegistry(models map[string]*Model, def string) (*Registry, error) {
	if len(models) == 0 {
		return nil, errors.New("registry needs at least one model")
	}
	if def == "" {
		def = sortedKeys(models)[0]
	}
	if _, ok := models[def]; !ok {
		return nil, fmt.Errorf("%w: default %q", ErrUnknownVersion, def)
	}
	return &Registry{models: models, Default: def}, nil
}

// RegistryFromSpec parses "v1,v2" or "v1:1234567,v2-canary:99", where the
// optional :seed gives a version its own weights.
func RegistryFromSpec(spec, def string) (*Registry, error) {
	models := map[string]*Model{}
	for _, entry := range strings.Split(spec, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		name, rawSeed, hasSeed := strings.Cut(entry, ":")
		name = strings.TrimSpace(name)
		seed := DefaultSeed
		if hasSeed {
			parsed, err := strconv.ParseInt(strings.TrimSpace(rawSeed), 10, 64)
			if err != nil {
				return nil, fmt.Errorf("bad seed for version %q: %w", name, err)
			}
			seed = parsed
		}
		models[name] = NewWithSeed(name, seed)
	}
	if len(models) == 0 {
		models["v1"] = New("v1")
	}
	return NewRegistry(models, def)
}

// Get resolves a pin; the empty string means the default version.
func (r *Registry) Get(version string) (*Model, error) {
	if version == "" {
		version = r.Default
	}
	m, ok := r.models[version]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownVersion, version)
	}
	return m, nil
}

// Versions lists loaded versions in stable order (used by /metrics and errors).
func (r *Registry) Versions() []string { return sortedKeys(r.models) }

// WarmupAll warms every resident version: a canary that is cold when it starts
// taking traffic looks exactly like a latency regression.
func (r *Registry) WarmupAll() {
	for _, m := range r.models {
		m.Warmup()
	}
}

func sortedKeys(m map[string]*Model) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

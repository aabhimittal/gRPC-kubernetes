// Package inference holds the model stand-in. A real backend (ONNX Runtime,
// Triton, a cgo call into libtorch) implements the same PredictBatch signature.
package inference

import "math"

const (
	FeatureDim = 16
	NumClasses = 4
)

// Model mirrors python/model.py: softmax(W·x) with seeded weights so both
// implementations produce comparable outputs.
type Model struct {
	Version string
	w       [NumClasses][FeatureDim]float64
}

func New(version string) *Model {
	m := &Model{Version: version}
	seed := int64(1234567)
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

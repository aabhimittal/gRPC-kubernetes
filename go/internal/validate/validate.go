// Package validate rejects requests that can never be served, before they
// reach the batcher. Malformed payloads are routine in production — a
// truncated vector from a lossy edge link, NaN from a sensor that lost signal,
// an oversized vector from a client bug — and cheap rejection keeps one bad
// client from spending everyone else's latency budget.
package validate

import (
	"fmt"
	"math"
)

// MaxFeatures bounds a single request regardless of model dimension, so a
// hostile or buggy client cannot force a large allocation.
const MaxFeatures = 4096

// Features checks one feature vector against the model's expected dimension.
// The returned error is safe to hand to an operator: it says what was wrong.
func Features(features []float32, dim int) error {
	switch n := len(features); {
	case n == 0:
		return fmt.Errorf("features must not be empty")
	case n > MaxFeatures:
		return fmt.Errorf("features length %d exceeds limit %d", n, MaxFeatures)
	case n != dim:
		return fmt.Errorf("features length %d does not match model dimension %d", n, dim)
	}
	for i, v := range features {
		f := float64(v)
		if math.IsNaN(f) {
			return fmt.Errorf("features[%d] is NaN", i)
		}
		if math.IsInf(f, 0) {
			return fmt.Errorf("features[%d] is infinite", i)
		}
	}
	return nil
}

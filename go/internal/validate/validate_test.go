package validate

import (
	"math"
	"strings"
	"testing"
)

func vec(n int) []float32 { return make([]float32, n) }

func TestAcceptsExactDimension(t *testing.T) {
	if err := Features(vec(16), 16); err != nil {
		t.Fatalf("valid vector rejected: %v", err)
	}
}

func TestRejectsMalformed(t *testing.T) {
	nan := vec(16)
	nan[0] = float32(math.NaN())
	inf := vec(16)
	inf[15] = float32(math.Inf(-1))

	cases := []struct {
		name     string
		features []float32
		dim      int
		want     string
	}{
		{"empty", nil, 16, "empty"},
		{"truncated", vec(15), 16, "does not match"},
		{"extra field", vec(17), 16, "does not match"},
		{"oversized", vec(MaxFeatures + 1), MaxFeatures + 1, "exceeds limit"},
		{"nan", nan, 16, "features[0] is NaN"},
		{"infinite", inf, 16, "features[15] is infinite"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := Features(tc.features, tc.dim)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want error containing %q", err, tc.want)
			}
		})
	}
}

func TestExtremeButFiniteValuesAllowed(t *testing.T) {
	f := vec(16)
	f[0], f[1] = math.MaxFloat32, -math.MaxFloat32
	f[2] = math.SmallestNonzeroFloat32
	if err := Features(f, 16); err != nil {
		t.Fatalf("legitimate extremes rejected: %v", err)
	}
}

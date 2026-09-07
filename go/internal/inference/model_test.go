package inference

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func row(v float32) []float32 {
	f := make([]float32, FeatureDim)
	for i := range f {
		f[i] = v
	}
	return f
}

func TestScoresAreAProbabilityDistribution(t *testing.T) {
	out := New("v1").PredictBatch([][]float32{row(0.5)})
	if len(out[0]) != NumClasses {
		t.Fatalf("got %d classes, want %d", len(out[0]), NumClasses)
	}
	var sum float64
	for _, s := range out[0] {
		if s < 0 || s > 1 {
			t.Fatalf("score %v outside [0,1]", s)
		}
		sum += float64(s)
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("scores sum to %v, want 1", sum)
	}
}

func TestBatchingDoesNotChangeResults(t *testing.T) {
	m := New("v1")
	rows := [][]float32{row(0), row(1), row(-2.5), row(0.125)}
	batched := m.PredictBatch(rows)
	for i, r := range rows {
		single := m.PredictBatch([][]float32{r})[0]
		if !reflect.DeepEqual(single, batched[i]) {
			t.Fatalf("row %d: batched %v != single %v", i, batched[i], single)
		}
	}
}

func TestEmptyBatchIsANoOp(t *testing.T) {
	if got := New("v1").PredictBatch(nil); len(got) != 0 {
		t.Fatalf("got %v, want empty", got)
	}
}

func TestLargeMagnitudeFeaturesDoNotOverflowSoftmax(t *testing.T) {
	out := New("v1").PredictBatch([][]float32{row(1e6), row(-1e6)})
	for _, scores := range out {
		var sum float64
		for _, s := range scores {
			if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				t.Fatalf("softmax produced %v", s)
			}
			sum += float64(s)
		}
		if math.Abs(sum-1) > 1e-6 {
			t.Fatalf("scores sum to %v, want 1", sum)
		}
	}
}

func TestShortVectorIsZeroPaddedNotPanicking(t *testing.T) {
	// Validation rejects these before the model sees them; the model still
	// must not panic if a future caller skips validation.
	if out := New("v1").PredictBatch([][]float32{{1, 2, 3}}); len(out[0]) != NumClasses {
		t.Fatal("short vector mishandled")
	}
}

func TestRegistryPinsAndDefaults(t *testing.T) {
	reg, err := RegistryFromSpec("v1:1234567,v2-canary:987654", "v1")
	if err != nil {
		t.Fatal(err)
	}
	if got := reg.Versions(); !reflect.DeepEqual(got, []string{"v1", "v2-canary"}) {
		t.Fatalf("versions = %v", got)
	}
	m, err := reg.Get("")
	if err != nil || m.Version != "v1" {
		t.Fatalf("empty pin -> %v, %v", m, err)
	}
	if m, err = reg.Get("v2-canary"); err != nil || m.Version != "v2-canary" {
		t.Fatalf("explicit pin -> %v, %v", m, err)
	}
}

func TestRegistryVersionsAreDifferentModels(t *testing.T) {
	reg, _ := RegistryFromSpec("v1:1234567,v2:987654", "v1")
	a, _ := reg.Get("v1")
	b, _ := reg.Get("v2")
	if reflect.DeepEqual(a.PredictBatch([][]float32{row(0.5)}), b.PredictBatch([][]float32{row(0.5)})) {
		t.Fatal("distinct seeds produced identical models")
	}
}

func TestUnknownVersionAndBadSpec(t *testing.T) {
	reg, _ := RegistryFromSpec("v1", "")
	if _, err := reg.Get("nope"); !errors.Is(err, ErrUnknownVersion) {
		t.Fatalf("got %v, want ErrUnknownVersion", err)
	}
	if _, err := RegistryFromSpec("v1", "v2"); !errors.Is(err, ErrUnknownVersion) {
		t.Fatalf("default not loaded: got %v", err)
	}
	if _, err := RegistryFromSpec("v1:not-a-number", ""); err == nil {
		t.Fatal("bad seed accepted")
	}
}

func TestSpecWithBlanksAndTrailingCommas(t *testing.T) {
	reg, err := RegistryFromSpec(" v1 , , v2 ,", "")
	if err != nil {
		t.Fatal(err)
	}
	if got := reg.Versions(); !reflect.DeepEqual(got, []string{"v1", "v2"}) {
		t.Fatalf("versions = %v", got)
	}
}

// TestCrossLanguageParity pins the Go and Python backends to the same answers.
// Both sit behind one Service, so a client must not be able to tell which one
// served it.
func TestCrossLanguageParity(t *testing.T) {
	var fixture struct {
		Version string `json:"version"`
		Seed    int64  `json:"seed"`
		Cases   []struct {
			Features []float32 `json:"features"`
			Scores   []float32 `json:"scores"`
		} `json:"cases"`
	}
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "testdata", "parity.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	m := NewWithSeed(fixture.Version, fixture.Seed)
	for i, c := range fixture.Cases {
		got := m.PredictBatch([][]float32{c.Features})[0]
		for j := range got {
			if math.Abs(float64(got[j]-c.Scores[j])) > 1e-6 {
				t.Fatalf("case %d class %d: go %v vs python %v", i, j, got[j], c.Scores[j])
			}
		}
	}
}

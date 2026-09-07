"""Model + multi-version registry behaviour, including cross-language parity."""
import json
import os

import pytest

from model import FEATURE_DIM, NUM_CLASSES, Model, ModelRegistry, UnknownModelVersion

PARITY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "parity.json"
)


def test_scores_are_a_probability_distribution():
    out = Model("v1").predict_batch([[0.1 * i for i in range(FEATURE_DIM)]])
    assert len(out[0]) == NUM_CLASSES
    assert abs(sum(out[0]) - 1.0) < 1e-9
    assert all(0.0 <= s <= 1.0 for s in out[0])


def test_batching_does_not_change_results():
    """Batch size is an implementation detail; a batched answer must equal the
    single-request answer bit-for-bit, or batching becomes a correctness bug."""
    m = Model("v1")
    rows = [[float(i + j) for i in range(FEATURE_DIM)] for j in range(7)]
    one_at_a_time = [m.predict_batch([r])[0] for r in rows]
    all_at_once = m.predict_batch(rows)
    assert one_at_a_time == all_at_once


def test_empty_batch_is_a_no_op():
    assert Model("v1").predict_batch([]) == []


def test_large_magnitude_features_do_not_overflow_softmax():
    """Softmax must subtract the max; otherwise exp() overflows to inf/NaN."""
    out = Model("v1").predict_batch([[1e6] * FEATURE_DIM, [-1e6] * FEATURE_DIM])
    for row in out:
        assert abs(sum(row) - 1.0) < 1e-9
        assert all(s == s for s in row)  # no NaN


def test_registry_pins_and_defaults():
    reg = ModelRegistry.from_spec("v1:1234567,v2-canary:987654", "v1")
    assert reg.versions() == ["v1", "v2-canary"]
    assert reg.get().version == "v1"           # empty pin -> default
    assert reg.get("v2-canary").version == "v2-canary"


def test_registry_versions_are_actually_different_models():
    reg = ModelRegistry.from_spec("v1:1234567,v2:987654", "v1")
    x = [[0.5] * FEATURE_DIM]
    assert reg.get("v1").predict_batch(x) != reg.get("v2").predict_batch(x)


def test_unknown_version_raises():
    reg = ModelRegistry.from_spec("v1")
    with pytest.raises(UnknownModelVersion):
        reg.get("does-not-exist")


def test_registry_rejects_default_it_does_not_hold():
    with pytest.raises(UnknownModelVersion):
        ModelRegistry({"v1": Model("v1")}, "v2")


def test_spec_with_blanks_and_trailing_commas():
    reg = ModelRegistry.from_spec(" v1 , , v2 ,")
    assert reg.versions() == ["v1", "v2"]


def test_cross_language_parity_fixture():
    """The Go and Python servers must agree: the same request routed to either
    backend behind one Service has to produce the same answer."""
    with open(PARITY) as fh:
        fixture = json.load(fh)
    m = Model(fixture["version"], fixture["seed"])
    for case in fixture["cases"]:
        got = m.predict_batch([case["features"]])[0]
        for g, want in zip(got, case["scores"]):
            assert abs(g - want) < 1e-6

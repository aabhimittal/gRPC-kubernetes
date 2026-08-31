"""Edge cases a real client will eventually send."""
import math

import pytest

from validation import MAX_FEATURES, InvalidInput, validate_features


def test_accepts_exact_dimension():
    validate_features([0.0] * 16, 16)  # must not raise


@pytest.mark.parametrize(
    "features, dim, expect",
    [
        ([], 16, "empty"),                                   # zero-length vector
        ([0.0] * 15, 16, "does not match"),                  # truncated payload
        ([0.0] * 17, 16, "does not match"),                  # extra field appended
        ([float("nan")] + [0.0] * 15, 16, "NaN"),            # dead sensor
        ([float("inf")] + [0.0] * 15, 16, "infinite"),       # divide-by-zero upstream
        ([float("-inf")] + [0.0] * 15, 16, "infinite"),
    ],
)
def test_rejects_malformed(features, dim, expect):
    with pytest.raises(InvalidInput) as err:
        validate_features(features, dim)
    assert expect in str(err.value)


def test_oversized_vector_rejected_before_dim_check():
    """A hostile 10 MB vector must be refused on size, not on dimension —
    the size check is what bounds memory."""
    with pytest.raises(InvalidInput) as err:
        validate_features([0.0] * (MAX_FEATURES + 1), MAX_FEATURES + 1)
    assert "exceeds limit" in str(err.value)


def test_nan_detected_at_any_position():
    features = [0.0] * 16
    features[15] = math.nan
    with pytest.raises(InvalidInput) as err:
        validate_features(features, 16)
    assert "features[15]" in str(err.value)


def test_denormal_and_extreme_but_finite_values_are_allowed():
    """Legitimate extremes must pass — over-strict validation drops good traffic."""
    validate_features([1e-320, -3.4e38, 3.4e38] + [0.0] * 13, 16)

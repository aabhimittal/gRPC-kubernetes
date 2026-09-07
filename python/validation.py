"""Request validation — the first line of defence for an inference endpoint.

Industrial callers send malformed payloads far more often than the happy path
suggests: truncated feature vectors from a lossy edge link, NaN from a sensor
that lost its signal, or a multi-megabyte vector from a client bug. Rejecting
those cheaply (before they reach the batcher) keeps one bad client from
degrading latency for everyone.
"""
from __future__ import annotations

import math
from typing import Sequence

# A single request may not exceed this many features regardless of model dim.
# Guards against memory blow-ups from a hostile or buggy client.
MAX_FEATURES = 4096


class InvalidInput(ValueError):
    """Raised for a request that can never be served — maps to INVALID_ARGUMENT."""


def validate_features(features: Sequence[float], dim: int) -> None:
    """Validate one feature vector against the model's expected dimension.

    Raises InvalidInput with an operator-readable reason. Never mutates input.
    """
    n = len(features)
    if n == 0:
        raise InvalidInput("features must not be empty")
    if n > MAX_FEATURES:
        raise InvalidInput(f"features length {n} exceeds limit {MAX_FEATURES}")
    if n != dim:
        raise InvalidInput(f"features length {n} does not match model dimension {dim}")
    for i, v in enumerate(features):
        if math.isnan(v):
            raise InvalidInput(f"features[{i}] is NaN")
        if math.isinf(v):
            raise InvalidInput(f"features[{i}] is infinite")

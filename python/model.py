"""Deterministic stand-in model.

Real backends (ONNX Runtime, PyTorch, Triton) implement the same
`predict_batch(matrix) -> matrix` interface; swap this class out and the server
is unchanged.
"""
from __future__ import annotations

import math
from typing import List

FEATURE_DIM = 16
NUM_CLASSES = 4


DEFAULT_SEED = 1234567


def _seeded_weights(seed: int = DEFAULT_SEED) -> List[List[float]]:
    # Fixed pseudo-random weights (no numpy dependency, reproducible across
    # languages). Mirrors go/internal/inference/model.go.
    w = [[0.0] * FEATURE_DIM for _ in range(NUM_CLASSES)]
    for c in range(NUM_CLASSES):
        for f in range(FEATURE_DIM):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            w[c][f] = (seed / 0x7FFFFFFF) * 2.0 - 1.0
    return w


class Model:
    #: Feature vector width this model accepts. Validation rejects anything else.
    dim = FEATURE_DIM

    def __init__(self, version: str = "v1", seed: int = DEFAULT_SEED) -> None:
        self.version = version
        self._w = _seeded_weights(seed)

    def predict_batch(self, batch: List[List[float]]) -> List[List[float]]:
        """batch: (n, FEATURE_DIM) -> (n, NUM_CLASSES) softmax scores."""
        out: List[List[float]] = []
        for feats in batch:
            f = (list(feats) + [0.0] * FEATURE_DIM)[:FEATURE_DIM]
            logits = [sum(self._w[c][i] * f[i] for i in range(FEATURE_DIM))
                      for c in range(NUM_CLASSES)]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            s = sum(exps)
            out.append([e / s for e in exps])
        return out

    def warmup(self) -> None:
        self.predict_batch([[0.0] * FEATURE_DIM])


class UnknownModelVersion(KeyError):
    """Requested model_version is not loaded — maps to NOT_FOUND."""


class ModelRegistry:
    """Holds several model versions at once so a rollout is a routing change.

    Canary and shadow deployments both need two versions resident in the same
    process: pinning `model_version` on the request routes to a specific one,
    while an empty pin follows `default_version`. Promoting a canary is then a
    one-line config flip rather than a redeploy, and rolling back is instant.
    """

    def __init__(self, versions: "dict[str, Model] | None" = None,
                 default_version: str = "v1") -> None:
        self._models: "dict[str, Model]" = dict(versions or {})
        if not self._models:
            self._models[default_version] = Model(default_version)
        if default_version not in self._models:
            raise UnknownModelVersion(default_version)
        self.default_version = default_version

    @classmethod
    def from_spec(cls, spec: str, default_version: str = "") -> "ModelRegistry":
        """Build from a comma-separated spec, e.g. "v1,v2" or "v1:1234567,v2:99".

        The optional `:seed` gives each version distinct weights so a canary
        actually behaves differently from the incumbent in the demo.
        """
        models: "dict[str, Model]" = {}
        for entry in (e.strip() for e in spec.split(",")):
            if not entry:
                continue
            name, _, seed = entry.partition(":")
            models[name] = Model(name, int(seed) if seed else DEFAULT_SEED)
        if not models:
            models["v1"] = Model("v1")
        default = default_version or next(iter(models))
        return cls(models, default)

    def get(self, version: str = "") -> Model:
        """Resolve a pin. Empty string means the default version."""
        name = version or self.default_version
        try:
            return self._models[name]
        except KeyError:
            raise UnknownModelVersion(name) from None

    def versions(self) -> "list[str]":
        return sorted(self._models)

    def __contains__(self, version: str) -> bool:
        return version in self._models

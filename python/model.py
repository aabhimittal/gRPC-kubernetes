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


def _seeded_weights() -> List[List[float]]:
    # Fixed pseudo-random weights (no numpy dependency, reproducible across
    # languages). Mirrors go/internal/inference/model.go.
    w = [[0.0] * FEATURE_DIM for _ in range(NUM_CLASSES)]
    seed = 1234567
    for c in range(NUM_CLASSES):
        for f in range(FEATURE_DIM):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            w[c][f] = (seed / 0x7FFFFFFF) * 2.0 - 1.0
    return w


class Model:
    def __init__(self, version: str = "v1") -> None:
        self.version = version
        self._w = _seeded_weights()

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

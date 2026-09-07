#!/usr/bin/env python3
"""Regenerate testdata/parity.json — the golden vectors both language test
suites check against.

Both backends sit behind one Service, so a client must not be able to tell
which one served it. Run this only when the model definition changes on
purpose; an unexpected diff here means the two implementations have drifted.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from model import DEFAULT_SEED, FEATURE_DIM, Model  # noqa: E402

CASES = [
    [0.0] * FEATURE_DIM,
    [1.0] * FEATURE_DIM,
    [0.1 * i for i in range(FEATURE_DIM)],
    [(-1.0) ** i * (i / 3.0) for i in range(FEATURE_DIM)],
    [1e3] * FEATURE_DIM,
]


def main() -> None:
    model = Model("v1", DEFAULT_SEED)
    fixture = {
        "_comment": (
            "Golden vectors shared by the Python and Go test suites. Both "
            "backends serve the same Service, so their outputs must agree "
            "within 1e-6. Regenerate with `make parity`."
        ),
        "version": "v1",
        "seed": DEFAULT_SEED,
        "feature_dim": FEATURE_DIM,
        "cases": [
            {"features": case, "scores": model.predict_batch([case])[0]} for case in CASES
        ],
    }
    out = os.path.join(ROOT, "testdata", "parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(fixture, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()

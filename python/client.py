"""Demo / load client. Shows unary, streaming, and concurrent batching.

Usage:
  python client.py --target localhost:50051 --mode unary
  python client.py --target localhost:50051 --mode stream --n 100
  python client.py --target localhost:50051 --mode load  --n 500 --concurrency 50
  python client.py --mode load --n 2000 --concurrency 400   # watch load shedding

The load mode is deliberately failure-tolerant: it tallies gRPC status codes
instead of dying on the first shed request, because the interesting runs are
exactly the ones where the server starts returning RESOURCE_EXHAUSTED or
DEADLINE_EXCEEDED.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import time

import grpc

from gen import inference_pb2 as pb
from gen import inference_pb2_grpc as pb_grpc
from metrics import Histogram

FEATURE_DIM = 16

# round_robin spreads streams across pods behind a headless Service; the retry
# policy turns the server's shed/drain signals into transparent retries. Only
# codes returned *before* the model runs are retried, so retries stay safe.
SERVICE_CONFIG = json.dumps({
    "loadBalancingConfig": [{"round_robin": {}}],
    "methodConfig": [{
        "name": [{"service": "inference.v1.InferenceService"}],
        "retryPolicy": {
            "maxAttempts": 4,
            "initialBackoff": "0.05s",
            "maxBackoff": "1s",
            "backoffMultiplier": 2,
            "retryableStatusCodes": ["UNAVAILABLE", "RESOURCE_EXHAUSTED"],
        },
    }],
})

CHANNEL_OPTIONS = [
    ("grpc.service_config", SERVICE_CONFIG),
    ("grpc.enable_retries", 1),
    # Keep the HTTP/2 connection alive across idle gaps so the next request
    # does not pay a fresh handshake.
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]


def _req(i: int, version: str = "") -> "pb.PredictRequest":
    return pb.PredictRequest(
        request_id=f"r{i}",
        features=[random.random() for _ in range(FEATURE_DIM)],
        model_version=version,
    )


async def unary(stub, version: str, timeout: float) -> None:
    resp = await stub.Predict(_req(0, version), timeout=timeout)
    print(f"label={resp.label} version={resp.model_version} batch_size={resp.batch_size} "
          f"latency={resp.server_latency_ms:.2f}ms scores={list(resp.scores)}")


async def stream(stub, n: int, version: str, timeout: float) -> None:
    async def gen():
        for i in range(n):
            yield _req(i, version)

    seen, max_batch = 0, 0
    async for resp in stub.PredictStream(gen(), timeout=timeout):
        seen += 1
        max_batch = max(max_batch, resp.batch_size)
    print(f"streamed {seen} predictions (max server batch_size={max_batch})")


async def load(stub, n: int, concurrency: int, version: str, timeout: float) -> None:
    sem = asyncio.Semaphore(concurrency)
    latency = Histogram(reservoir=max(n, 1))
    codes: "collections.Counter" = collections.Counter()
    batch_sizes: list[int] = []

    async def one(i: int) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                resp = await stub.Predict(_req(i, version), timeout=timeout)
            except grpc.aio.AioRpcError as exc:
                codes[exc.code().name] += 1     # shedding is a valid outcome
                return
            latency.observe((time.monotonic() - t0) * 1000.0)
            codes["OK"] += 1
            batch_sizes.append(resp.batch_size)

    start = time.monotonic()
    await asyncio.gather(*(one(i) for i in range(n)))
    elapsed = time.monotonic() - start
    avg = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
    print(f"sent {n} @ concurrency {concurrency} in {elapsed:.2f}s "
          f"({n / elapsed:.0f} rps)")
    print(f"  codes={dict(sorted(codes.items()))} avg server batch_size={avg:.1f} "
          f"(>1 means batching kicked in)")
    print(f"  client latency p50={latency.percentile(0.5):.1f}ms "
          f"p95={latency.percentile(0.95):.1f}ms p99={latency.percentile(0.99):.1f}ms")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="localhost:50051")
    ap.add_argument("--mode", choices=["unary", "stream", "load"], default="unary")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--version", default="", help="model_version pin (empty = server default)")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-RPC deadline in seconds")
    args = ap.parse_args()

    async with grpc.aio.insecure_channel(args.target, options=CHANNEL_OPTIONS) as channel:
        stub = pb_grpc.InferenceServiceStub(channel)
        if args.mode == "unary":
            await unary(stub, args.version, args.timeout)
        elif args.mode == "stream":
            await stream(stub, args.n, args.version, args.timeout)
        else:
            await load(stub, args.n, args.concurrency, args.version, args.timeout)


if __name__ == "__main__":
    asyncio.run(main())

"""Demo / load client. Shows unary, streaming, and concurrent batching.

Usage:
  python client.py --target localhost:50051 --mode unary
  python client.py --target localhost:50051 --mode stream --n 100
  python client.py --target localhost:50051 --mode load  --n 500 --concurrency 50
"""
from __future__ import annotations

import argparse
import asyncio
import random

import grpc

from gen import inference_pb2 as pb
from gen import inference_pb2_grpc as pb_grpc

FEATURE_DIM = 16


def _req(i: int) -> "pb.PredictRequest":
    return pb.PredictRequest(
        request_id=f"r{i}",
        features=[random.random() for _ in range(FEATURE_DIM)],
    )


async def unary(stub) -> None:
    resp = await stub.Predict(_req(0))
    print(f"label={resp.label} batch_size={resp.batch_size} "
          f"latency={resp.server_latency_ms:.2f}ms scores={list(resp.scores)}")


async def stream(stub, n: int) -> None:
    async def gen():
        for i in range(n):
            yield _req(i)
    seen = 0
    async for resp in stub.PredictStream(gen()):
        seen += 1
    print(f"streamed {seen} predictions")


async def load(stub, n: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    batch_sizes = []

    async def one(i: int):
        async with sem:
            resp = await stub.Predict(_req(i))
            batch_sizes.append(resp.batch_size)

    await asyncio.gather(*(one(i) for i in range(n)))
    avg = sum(batch_sizes) / len(batch_sizes)
    print(f"sent {n} requests @ concurrency {concurrency}; "
          f"avg server batch_size={avg:.1f} (>1 means batching kicked in)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="localhost:50051")
    ap.add_argument("--mode", choices=["unary", "stream", "load"], default="unary")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()

    # round_robin spreads streams across pods behind a headless Service.
    async with grpc.aio.insecure_channel(
        args.target,
        options=[("grpc.service_config",
                  '{"loadBalancingConfig":[{"round_robin":{}}]}')],
    ) as channel:
        stub = pb_grpc.InferenceServiceStub(channel)
        if args.mode == "unary":
            await unary(stub)
        elif args.mode == "stream":
            await stream(stub, args.n)
        else:
            await load(stub, args.n, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())

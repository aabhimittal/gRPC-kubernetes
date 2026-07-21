"""Async gRPC inference server with dynamic batching."""
from __future__ import annotations

import asyncio
import os
import time

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from batcher import DynamicBatcher
from model import Model

# Generated at build time into python/gen (see Makefile / Dockerfile).
from gen import inference_pb2 as pb
from gen import inference_pb2_grpc as pb_grpc


class InferenceServicer(pb_grpc.InferenceServiceServicer):
    def __init__(self, model: Model, batcher: DynamicBatcher) -> None:
        self._model = model
        self._batcher = batcher
        self._ready = False

    def set_ready(self, v: bool) -> None:
        self._ready = v

    async def Predict(self, request, context):
        start = time.monotonic()
        res = await self._batcher.submit(list(request.features))
        return self._to_response(request.request_id, res, start)

    async def PredictStream(self, request_iterator, context):
        async for request in request_iterator:
            start = time.monotonic()
            res = await self._batcher.submit(list(request.features))
            yield self._to_response(request.request_id, res, start)

    async def HealthCheck(self, request, context):
        status = pb.HealthResponse.SERVING if self._ready else pb.HealthResponse.NOT_SERVING
        return pb.HealthResponse(status=status, model_version=self._model.version)

    def _to_response(self, request_id, res, start):
        scores = res.scores
        label = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        return pb.PredictResponse(
            request_id=request_id,
            scores=scores,
            label=label,
            model_version=self._model.version,
            server_latency_ms=(time.monotonic() - start) * 1000.0,
            batch_size=res.batch_size,
        )


async def serve() -> None:
    port = os.getenv("PORT", "50051")
    model = Model(os.getenv("MODEL_VERSION", "v1"))
    batcher = DynamicBatcher(
        model,
        max_batch=int(os.getenv("MAX_BATCH", "32")),
        max_delay_ms=int(os.getenv("MAX_DELAY_MS", "5")),
    )

    server = grpc.aio.server()
    servicer = InferenceServicer(model, batcher)
    pb_grpc.add_InferenceServiceServicer_to_server(servicer, server)

    # Standard grpc.health.v1 service — queried by Kubernetes native gRPC
    # probes / grpc_health_probe. NOT_SERVING until warm-up completes.
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)

    server.add_insecure_port(f"[::]:{port}")

    await server.start()
    # Warm-up before advertising readiness (docs/LLD.md §6).
    model.warmup()
    batcher.start()
    servicer.set_ready(True)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    print(f"python inference server listening on :{port}", flush=True)

    async def shutdown() -> None:
        await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        await batcher.stop()
        await server.stop(grace=10)

    try:
        await server.wait_for_termination()
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(serve())

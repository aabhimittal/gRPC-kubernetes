"""Async gRPC inference server with dynamic batching.

Wiring beyond the happy path (see docs/LLD.md §7):
  * per-request validation with precise gRPC status codes,
  * client deadline propagated into the batcher so doomed work is dropped,
  * admission control surfaced as RESOURCE_EXHAUSTED, not a latency cliff,
  * model-version pinning for canary/shadow rollouts,
  * pipelined bidirectional streaming (many in-flight per stream),
  * keepalive tuned to survive idle L4 load balancers,
  * Prometheus metrics on a side port,
  * draining shutdown that fails queued work fast instead of hanging clients.
"""
from __future__ import annotations

import asyncio
import os
import time

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

import metrics as metrics_mod
from batcher import DynamicBatcher, Expired, QueueFull, ShuttingDown
from model import ModelRegistry, UnknownModelVersion
from validation import InvalidInput, validate_features

# Generated at build time into python/gen (see Makefile / Dockerfile).
from gen import inference_pb2 as pb
from gen import inference_pb2_grpc as pb_grpc


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "")) if os.getenv(key) else default
    except ValueError:
        return default


class InferenceServicer(pb_grpc.InferenceServiceServicer):
    def __init__(
        self,
        registry: ModelRegistry,
        batcher: DynamicBatcher,
        metrics: metrics_mod.Registry,
        stream_concurrency: int = 32,
    ) -> None:
        self._registry = registry
        self._batcher = batcher
        self._metrics = metrics
        self._stream_concurrency = max(1, stream_concurrency)
        self._ready = False

    def set_ready(self, v: bool) -> None:
        self._ready = v

    # ---- RPCs ------------------------------------------------------------
    async def Predict(self, request, context):
        start = time.monotonic()
        self._metrics.inc("inference_requests_total")
        try:
            res = await self._predict(request, context)
        except _Rejected as rej:
            await self._abort(context, rej)
            return pb.PredictResponse()  # unreachable: abort raises
        self._metrics.observe("inference_latency_ms", (time.monotonic() - start) * 1000.0)
        self._metrics.inc("inference_responses_total")
        return self._to_response(request.request_id, res, start)

    async def PredictStream(self, request_iterator, context):
        """Pipelined: requests are dispatched concurrently and responses are
        emitted as they complete, so one slow item cannot stall the stream and
        a single client can actually fill a batch. Responses may be reordered —
        correlate on `request_id`, which is what it is for."""
        out: "asyncio.Queue" = asyncio.Queue()
        sem = asyncio.Semaphore(self._stream_concurrency)
        inflight: "set[asyncio.Task]" = set()
        reader_done = asyncio.Event()

        async def handle(request) -> None:
            start = time.monotonic()
            self._metrics.inc("inference_stream_requests_total")
            try:
                async with sem:
                    res = await self._predict(request, context)
                self._metrics.observe(
                    "inference_latency_ms", (time.monotonic() - start) * 1000.0
                )
                await out.put(("ok", self._to_response(request.request_id, res, start)))
            except _Rejected as rej:
                await out.put(("err", rej))
            except Exception as exc:  # pragma: no cover - defensive
                await out.put(("err", _Rejected(grpc.StatusCode.INTERNAL, str(exc))))

        async def reader() -> None:
            try:
                async for request in request_iterator:
                    task = asyncio.create_task(handle(request))
                    inflight.add(task)
                    task.add_done_callback(inflight.discard)
            finally:
                reader_done.set()

        reader_task = asyncio.create_task(reader())
        try:
            while True:
                if reader_done.is_set() and not inflight and out.empty():
                    break
                try:
                    kind, payload = await asyncio.wait_for(out.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if kind == "err":
                    await self._abort(context, payload)
                    return
                self._metrics.inc("inference_stream_responses_total")
                yield payload
        finally:
            reader_task.cancel()
            for task in list(inflight):
                task.cancel()

    async def HealthCheck(self, request, context):
        status = pb.HealthResponse.SERVING if self._ready else pb.HealthResponse.NOT_SERVING
        return pb.HealthResponse(
            status=status, model_version=self._registry.default_version
        )

    # ---- internals -------------------------------------------------------
    async def _predict(self, request, context):
        try:
            model = self._registry.get(request.model_version)
        except UnknownModelVersion:
            self._metrics.inc("inference_errors_total")
            raise _Rejected(
                grpc.StatusCode.NOT_FOUND,
                f"unknown model_version {request.model_version!r}; "
                f"loaded: {','.join(self._registry.versions())}",
            ) from None
        try:
            validate_features(request.features, model.dim)
        except InvalidInput as exc:
            self._metrics.inc("inference_invalid_total")
            raise _Rejected(grpc.StatusCode.INVALID_ARGUMENT, str(exc)) from None

        remaining = context.time_remaining()
        deadline = time.monotonic() + remaining if remaining is not None else None
        try:
            return await self._batcher.submit(
                list(request.features), request.model_version, deadline
            )
        except QueueFull as exc:
            self._metrics.inc("inference_rejected_total")
            raise _Rejected(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc)) from None
        except Expired as exc:
            self._metrics.inc("inference_expired_total")
            raise _Rejected(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc)) from None
        except ShuttingDown as exc:
            raise _Rejected(grpc.StatusCode.UNAVAILABLE, str(exc)) from None

    @staticmethod
    async def _abort(context, rejected: "_Rejected"):
        await context.abort(rejected.code, rejected.details)

    def _to_response(self, request_id, res, start):
        scores = res.scores
        label = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        return pb.PredictResponse(
            request_id=request_id,
            scores=scores,
            label=label,
            model_version=res.model_version,
            server_latency_ms=(time.monotonic() - start) * 1000.0,
            batch_size=res.batch_size,
        )


class _Rejected(Exception):
    """Internal carrier for (gRPC status code, message)."""

    def __init__(self, code, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


def build_server(registry, batcher, metrics, port: str):
    """Server options matter as much as the handlers in a k8s deployment."""
    options = [
        # Survive idle timeouts on L4 load balancers / NAT without a fresh
        # TCP+TLS handshake on the next request.
        ("grpc.keepalive_time_ms", _env_int("KEEPALIVE_MS", 30_000)),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
        # Bound memory per message; the batcher bounds concurrency separately.
        ("grpc.max_receive_message_length", _env_int("MAX_MSG_BYTES", 4 * 1024 * 1024)),
        ("grpc.max_send_message_length", _env_int("MAX_MSG_BYTES", 4 * 1024 * 1024)),
        # Force clients off this pod periodically so a scale-out actually
        # rebalances long-lived HTTP/2 connections (docs/LLD.md §5).
        ("grpc.max_connection_age_ms", _env_int("MAX_CONN_AGE_MS", 300_000)),
        ("grpc.max_connection_age_grace_ms", 10_000),
    ]
    server = grpc.aio.server(options=options)
    servicer = InferenceServicer(
        registry, batcher, metrics, _env_int("STREAM_CONCURRENCY", 32)
    )
    pb_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server, servicer


async def serve() -> None:
    port = os.getenv("PORT", "50051")
    registry = ModelRegistry.from_spec(
        os.getenv("MODEL_VERSIONS", os.getenv("MODEL_VERSION", "v1")),
        os.getenv("MODEL_VERSION", ""),
    )
    metrics = metrics_mod.Registry()
    batcher = DynamicBatcher(
        registry,
        max_batch=_env_int("MAX_BATCH", 32),
        max_delay_ms=_env_int("MAX_DELAY_MS", 5),
        max_queue=_env_int("MAX_QUEUE", 1024),
        min_delay_ms=_env_int("MIN_DELAY_US", 200) / 1000.0,
        adaptive=os.getenv("ADAPTIVE_BATCHING", "1") != "0",
        metrics=metrics,
    )

    server, servicer = build_server(registry, batcher, metrics, port)

    # Standard grpc.health.v1 service — queried by Kubernetes native gRPC
    # probes / grpc_health_probe. NOT_SERVING until warm-up completes.
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)

    metrics_srv = await metrics_mod.serve_metrics(metrics, _env_int("METRICS_PORT", 9090))

    await server.start()
    # Warm up every resident version before advertising readiness: a canary
    # that is cold when it starts taking traffic looks like a latency bug.
    for version in registry.versions():
        registry.get(version).warmup()
    batcher.start()
    servicer.set_ready(True)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    print(
        f"python inference server listening on :{port} "
        f"(metrics :{_env_int('METRICS_PORT', 9090)}, versions {registry.versions()})",
        flush=True,
    )

    async def shutdown() -> None:
        # Fail readiness first so the endpoint is pulled from Service
        # endpoints before we stop accepting work (docs/LLD.md §6).
        await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        servicer.set_ready(False)
        await asyncio.sleep(_env_int("PRESTOP_SLEEP_MS", 0) / 1000.0)
        await server.stop(grace=_env_int("GRACE_MS", 10_000) / 1000.0)
        await batcher.stop()
        metrics_srv.close()

    try:
        await server.wait_for_termination()
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(serve())

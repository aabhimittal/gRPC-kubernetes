"""End-to-end tests over a real gRPC connection.

These assert the status codes a client actually has to branch on — the part of
a serving tier that most often disappoints its callers — and are skipped when
the generated stubs are absent (run `make gen-python` first).
"""
import asyncio
import math

import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("gen.inference_pb2", reason="run `make gen-python` first")

from batcher import DynamicBatcher                     # noqa: E402
from gen import inference_pb2 as pb                    # noqa: E402
from gen import inference_pb2_grpc as pb_grpc          # noqa: E402
from metrics import Registry as MetricsRegistry        # noqa: E402
from model import FEATURE_DIM, ModelRegistry           # noqa: E402
from server import InferenceServicer                   # noqa: E402

FEATURES = [0.5] * FEATURE_DIM


class Harness:
    def __init__(self, servicer, batcher, channel, stub):
        self.servicer, self.batcher = servicer, batcher
        self.channel, self.stub = channel, stub


async def start(spec="v1", start_loop=True, **batcher_kwargs):
    registry = ModelRegistry.from_spec(spec)
    metrics = MetricsRegistry()
    batcher = DynamicBatcher(registry, metrics=metrics, **batcher_kwargs)
    if start_loop:
        batcher.start()
    servicer = InferenceServicer(registry, batcher, metrics)
    servicer.set_ready(True)

    server = grpc.aio.server()
    pb_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    return server, Harness(servicer, batcher, channel, pb_grpc.InferenceServiceStub(channel))


def scenario(setup, check):
    """Run `setup()` -> (server, harness), await `check(harness)`, clean up."""
    async def main():
        server, harness = await setup()
        try:
            await check(harness)
        finally:
            await harness.channel.close()
            await harness.batcher.stop(drain_timeout=1.0)
            await server.stop(grace=0)

    asyncio.run(main())


def test_predict_happy_path():
    async def check(h):
        resp = await h.stub.Predict(pb.PredictRequest(request_id="r1", features=FEATURES))
        assert resp.request_id == "r1"
        assert resp.model_version == "v1"
        assert resp.batch_size >= 1
        assert abs(sum(resp.scores) - 1.0) < 1e-6
        assert 0 <= resp.label < len(resp.scores)

    scenario(lambda: start(max_delay_ms=5), check)


@pytest.mark.parametrize(
    "features",
    [[], [0.0] * (FEATURE_DIM - 1), [0.0] * (FEATURE_DIM + 1), [math.nan] + [0.0] * (FEATURE_DIM - 1)],
    ids=["empty", "truncated", "too-long", "nan"],
)
def test_invalid_requests_map_to_invalid_argument(features):
    async def check(h):
        with pytest.raises(grpc.aio.AioRpcError) as err:
            await h.stub.Predict(pb.PredictRequest(features=features))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    scenario(lambda: start(max_delay_ms=5), check)


def test_unknown_model_version_maps_to_not_found():
    async def check(h):
        with pytest.raises(grpc.aio.AioRpcError) as err:
            await h.stub.Predict(pb.PredictRequest(features=FEATURES, model_version="v99"))
        assert err.value.code() == grpc.StatusCode.NOT_FOUND
        assert "loaded" in err.value.details()

    scenario(lambda: start("v1,v2", max_delay_ms=5), check)


def test_version_pin_routes_to_that_model():
    async def check(h):
        a = await h.stub.Predict(pb.PredictRequest(features=FEATURES, model_version="v1"))
        b = await h.stub.Predict(pb.PredictRequest(features=FEATURES, model_version="v2"))
        assert (a.model_version, b.model_version) == ("v1", "v2")
        assert list(a.scores) != list(b.scores)

    scenario(lambda: start("v1:1,v2:2", max_delay_ms=5), check)


def test_overload_sheds_with_resource_exhausted():
    """Batcher loop is not started, so the queue fills: the server must shed
    load instead of letting latency grow without bound."""

    async def check(h):
        codes = set()
        pending = [
            asyncio.ensure_future(h.stub.Predict(pb.PredictRequest(features=FEATURES)))
            for _ in range(6)
        ]
        await asyncio.sleep(0.3)
        for call in pending:
            if call.done():
                try:
                    call.result()
                except grpc.aio.AioRpcError as exc:
                    codes.add(exc.code())
        for call in pending:
            call.cancel()
        assert grpc.StatusCode.RESOURCE_EXHAUSTED in codes

    scenario(lambda: start(start_loop=False, max_queue=1, max_delay_ms=5), check)


def test_stream_returns_every_request_and_is_pipelined():
    async def check(h):
        n = 24

        async def requests():
            for i in range(n):
                yield pb.PredictRequest(request_id=f"r{i}", features=[i / 10.0] * FEATURE_DIM)

        seen, max_batch = set(), 0
        async for resp in h.stub.PredictStream(requests()):
            assert resp.request_id not in seen, "duplicate response"
            seen.add(resp.request_id)
            max_batch = max(max_batch, resp.batch_size)
        assert len(seen) == n
        # A serialized stream would batch one request at a time.
        assert max_batch >= 2, "stream is not pipelined"

    scenario(lambda: start(max_batch=32, max_delay_ms=20), check)


def test_stream_surfaces_per_request_errors():
    async def check(h):
        async def requests():
            yield pb.PredictRequest(request_id="bad", features=[])

        with pytest.raises(grpc.aio.AioRpcError) as err:
            async for _ in h.stub.PredictStream(requests()):
                pass
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    scenario(lambda: start(max_delay_ms=5), check)


def test_health_check_reflects_readiness():
    async def check(h):
        resp = await h.stub.HealthCheck(pb.HealthRequest())
        assert resp.status == pb.HealthResponse.SERVING
        h.servicer.set_ready(False)
        resp = await h.stub.HealthCheck(pb.HealthRequest())
        assert resp.status == pb.HealthResponse.NOT_SERVING

    scenario(lambda: start(), check)


def test_concurrent_calls_are_coalesced():
    async def check(h):
        calls = [
            h.stub.Predict(pb.PredictRequest(request_id=f"c{i}", features=FEATURES))
            for i in range(40)
        ]
        responses = await asyncio.gather(*calls)
        assert len(responses) == 40
        assert max(r.batch_size for r in responses) >= 2

    scenario(lambda: start(max_batch=64, max_delay_ms=25), check)

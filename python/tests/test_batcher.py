"""Batcher behaviour under the conditions that actually break inference tiers:
overload, client timeouts, mixed model versions, model faults and shutdown.

Tests drive asyncio directly (no pytest-asyncio) to keep CI dependency-free.
"""
import asyncio
import time

import pytest

from batcher import DynamicBatcher, Expired, QueueFull, ShuttingDown
from metrics import Registry as MetricsRegistry
from model import FEATURE_DIM, Model, ModelRegistry, UnknownModelVersion

X = [0.5] * FEATURE_DIM


def run(coro):
    return asyncio.run(coro)


async def _with_batcher(b, body):
    b.start()
    try:
        return await body()
    finally:
        await b.stop(drain_timeout=1.0)


class _SlowModel(Model):
    def __init__(self, delay_s: float) -> None:
        super().__init__("slow")
        self._delay = delay_s

    def predict_batch(self, batch):
        time.sleep(self._delay)
        return super().predict_batch(batch)


class _BrokenModel(Model):
    def __init__(self) -> None:
        super().__init__("broken")
        self.calls = 0

    def predict_batch(self, batch):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return super().predict_batch(batch)


def test_concurrent_requests_coalesce_into_one_batch():
    async def body():
        results = await asyncio.gather(*(b.submit(X) for _ in range(8)))
        assert {r.batch_size for r in results} == {8}
        return results

    b = DynamicBatcher(Model("v1"), max_batch=32, max_delay_ms=25)
    run(_with_batcher(b, body))


def test_batch_never_exceeds_max_batch():
    async def body():
        results = await asyncio.gather(*(b.submit(X) for _ in range(20)))
        assert len(results) == 20
        assert max(r.batch_size for r in results) <= 4
        assert min(r.batch_size for r in results) >= 1

    b = DynamicBatcher(Model("v1"), max_batch=4, max_delay_ms=25)
    run(_with_batcher(b, body))


def test_single_request_flushes_on_the_delay_window():
    async def body():
        started = time.monotonic()
        res = await b.submit(X)
        assert res.batch_size == 1
        # Must not wait longer than the configured window (plus scheduling slack).
        assert (time.monotonic() - started) < 0.5

    b = DynamicBatcher(Model("v1"), max_batch=32, max_delay_ms=10, adaptive=False)
    run(_with_batcher(b, body))


def test_queue_full_rejects_immediately_instead_of_queueing_forever():
    """Admission control: past the queue depth, callers get a fast, retryable
    error rather than a request that sits behind a thousand others."""

    async def body():
        # Loop deliberately not started, so nothing drains the queue.
        first = asyncio.ensure_future(b.submit(X))
        second = asyncio.ensure_future(b.submit(X))
        await asyncio.sleep(0)
        with pytest.raises(QueueFull) as err:
            await b.submit(X)
        assert "queue full" in str(err.value)
        for task in (first, second):
            task.cancel()
        await b.stop(drain_timeout=0.2)

    b = DynamicBatcher(Model("v1"), max_batch=8, max_delay_ms=5, max_queue=2)
    run(body())


def test_deadline_already_passed_on_arrival_is_refused_without_queueing():
    async def body():
        with pytest.raises(Expired):
            await b.submit(X, deadline=time.monotonic() - 0.001)
        assert b.queue_depth == 0

    b = DynamicBatcher(Model("v1"), max_delay_ms=5)
    run(_with_batcher(b, body))


def test_work_whose_deadline_expired_while_queued_is_dropped_before_compute():
    """The whole point of deadline propagation: never spend model time on a
    request whose client has already given up."""
    model = _SlowModel(0.0)

    async def body():
        pending = asyncio.ensure_future(b.submit(X, deadline=time.monotonic() + 0.05))
        await asyncio.sleep(0.12)          # deadline lapses while the loop is idle
        b.start()                           # only now does the batcher run
        with pytest.raises(Expired):
            await pending
        await b.stop(drain_timeout=0.5)

    b = DynamicBatcher(model, max_batch=8, max_delay_ms=5)
    run(body())


def test_unknown_model_version_is_refused_before_taking_a_queue_slot():
    async def body():
        with pytest.raises(UnknownModelVersion):
            await b.submit(X, version="v9")
        assert b.queue_depth == 0

    b = DynamicBatcher(ModelRegistry.from_spec("v1,v2"), max_delay_ms=5)
    run(_with_batcher(b, body))


def test_mixed_versions_in_one_window_are_split_into_per_version_batches():
    """Canary traffic must not be blocked by — or blended into — the
    incumbent's batch."""

    async def body():
        results = await asyncio.gather(
            b.submit(X, "v1"), b.submit(X, "v1"), b.submit(X, "v1"),
            b.submit(X, "v2"), b.submit(X, "v2"),
        )
        by_version = {}
        for r in results:
            by_version.setdefault(r.model_version, []).append(r.batch_size)
        assert by_version == {"v1": [3, 3, 3], "v2": [2, 2]}

    b = DynamicBatcher(ModelRegistry.from_spec("v1:1,v2:2", "v1"),
                       max_batch=32, max_delay_ms=25)
    run(_with_batcher(b, body))


def test_adaptive_window_shrinks_under_load_and_recovers_when_idle():
    async def body():
        start_delay = b.current_delay_ms
        for _ in range(4):                       # saturate: every batch fills
            await asyncio.gather(*(b.submit(X) for _ in range(2)))
        assert b.current_delay_ms < start_delay
        low = b.current_delay_ms
        for _ in range(3):                       # idle: batches of one
            await b.submit(X)
        assert b.current_delay_ms > low

    b = DynamicBatcher(Model("v1"), max_batch=2, max_delay_ms=40, min_delay_ms=1)
    run(_with_batcher(b, body))


def test_adaptive_window_is_clamped_to_the_configured_bounds():
    async def body():
        for _ in range(50):
            await asyncio.gather(*(b.submit(X) for _ in range(2)))
        assert b.current_delay_ms >= 2.0
        for _ in range(50):
            await b.submit(X)
        assert b.current_delay_ms <= 20.0

    b = DynamicBatcher(Model("v1"), max_batch=2, max_delay_ms=20, min_delay_ms=2)
    run(_with_batcher(b, body))


def test_model_failure_fails_only_that_batch_and_the_loop_survives():
    model = _BrokenModel()

    async def body():
        with pytest.raises(RuntimeError):
            await b.submit(X)
        res = await b.submit(X)               # loop still alive
        assert res.batch_size == 1
        assert model.calls == 2

    b = DynamicBatcher(model, max_batch=1, max_delay_ms=5)
    run(_with_batcher(b, body))


def test_shutdown_fails_queued_work_fast_instead_of_hanging_clients():
    async def body():
        b.start()
        await b.submit(X)                     # loop is up
        queued = asyncio.ensure_future(b.submit(X))
        await asyncio.sleep(0)
        await b.stop(drain_timeout=1.0)
        try:
            await queued                      # either served during drain ...
        except ShuttingDown:
            pass                              # ... or failed fast. Never hangs.
        with pytest.raises(ShuttingDown):
            await b.submit(X)                 # closed to new work

    b = DynamicBatcher(Model("v1"), max_batch=8, max_delay_ms=5)
    run(body())


def test_stop_is_idempotent():
    async def body():
        b.start()
        await b.stop(drain_timeout=0.5)
        await b.stop(drain_timeout=0.5)

    b = DynamicBatcher(Model("v1"))
    run(body())


def test_cancelled_caller_does_not_break_the_batch_it_was_in():
    async def body():
        doomed = asyncio.ensure_future(b.submit(X))
        await asyncio.sleep(0)
        doomed.cancel()
        results = await asyncio.gather(*(b.submit(X) for _ in range(3)))
        assert all(r.batch_size >= 1 for r in results)

    b = DynamicBatcher(Model("v1"), max_batch=8, max_delay_ms=20)
    run(_with_batcher(b, body))


def test_metrics_are_recorded_for_batches_and_rejections():
    m = MetricsRegistry()

    async def body():
        await asyncio.gather(*(b.submit(X) for _ in range(4)))
        assert m.counter("inference_batches_total") >= 1
        assert m.counter("inference_batched_requests_total") == 4
        assert m.histogram("inference_batch_size").percentile(0.5) >= 1

    b = DynamicBatcher(Model("v1"), max_batch=8, max_delay_ms=20, metrics=m)
    run(_with_batcher(b, body))


@pytest.mark.parametrize(
    "kwargs", [{"max_batch": 0}, {"max_queue": 0}, {"min_delay_ms": 50, "max_delay_ms": 5}]
)
def test_invalid_configuration_fails_at_construction(kwargs):
    with pytest.raises(ValueError):
        DynamicBatcher(Model("v1"), **kwargs)

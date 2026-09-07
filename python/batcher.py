"""Async dynamic batcher — see docs/LLD.md §2.

Beyond plain coalescing, this batcher carries the controls a production
inference tier needs when traffic stops being polite:

* **Admission control** — a bounded queue. Past its depth new work is rejected
  immediately (RESOURCE_EXHAUSTED) instead of silently building an unbounded
  latency tail that outlives every client's deadline.
* **Deadline-aware batching** — queued work whose client deadline has already
  passed is dropped *before* the model call. Under a latency spike this is what
  stops the server burning GPU cycles on answers nobody is waiting for.
* **Adaptive batch window** — the wait shrinks toward `min_delay_ms` while
  batches keep filling (arrival rate is high; waiting buys nothing) and grows
  back toward `max_delay_ms` when traffic thins, so a quiet service does not
  pay a fixed 5 ms tax per request.
* **Version-aware grouping** — requests pinned to different model versions in
  one window are split into per-version model calls, so canary traffic never
  blocks the incumbent's batch.
* **Draining shutdown** — on stop, queued work is failed fast with a retryable
  error rather than left hanging until the client's own deadline fires.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from model import Model, ModelRegistry


class QueueFull(Exception):
    """Admission control rejected the request — maps to RESOURCE_EXHAUSTED."""


class Expired(Exception):
    """Deadline passed while queued — maps to DEADLINE_EXCEEDED."""


class ShuttingDown(Exception):
    """Server is draining — maps to UNAVAILABLE (safe for the client to retry)."""


@dataclass
class _Item:
    features: List[float]
    future: "asyncio.Future"
    version: str = ""
    deadline: Optional[float] = None  # time.monotonic() basis
    enqueued_at: float = field(default_factory=time.monotonic)

    def expired(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline


@dataclass
class Result:
    scores: List[float]
    batch_size: int
    model_version: str
    queue_wait_ms: float = 0.0


class DynamicBatcher:
    def __init__(
        self,
        model: "Model | ModelRegistry",
        max_batch: int = 32,
        max_delay_ms: int = 5,
        max_queue: int = 1024,
        min_delay_ms: float = 0.2,
        adaptive: bool = True,
        metrics=None,
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        if min_delay_ms > max_delay_ms:
            raise ValueError("min_delay_ms must not exceed max_delay_ms")
        self._registry = (
            model if isinstance(model, ModelRegistry)
            else ModelRegistry({model.version: model}, model.version)
        )
        self._max_batch = max_batch
        self._max_delay = max_delay_ms / 1000.0
        self._min_delay = min_delay_ms / 1000.0
        self._adaptive = adaptive
        self._delay = self._max_delay
        self._metrics = metrics
        self._queue: "asyncio.Queue[_Item]" = asyncio.Queue(maxsize=max_queue)
        self._task: "asyncio.Task | None" = None
        self._stopping = asyncio.Event()
        self._closed = False

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Refuse new work, let the loop finish what it holds, fail the rest."""
        self._closed = True
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), drain_timeout)
            except asyncio.TimeoutError:
                self._task.cancel()
            except asyncio.CancelledError:
                pass
            self._task = None
        self._reject_pending()

    def _reject_pending(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._fail(item, ShuttingDown("server is shutting down"))

    # ---- introspection (used by /metrics and tests) ----------------------
    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def current_delay_ms(self) -> float:
        return self._delay * 1000.0

    # ---- submit ----------------------------------------------------------
    async def submit(
        self,
        features: List[float],
        version: str = "",
        deadline: Optional[float] = None,
    ) -> Result:
        if self._closed:
            raise ShuttingDown("server is shutting down")
        # Resolve the pin up front: an unknown version must fail before it
        # consumes a queue slot.
        self._registry.get(version)
        if deadline is not None and deadline <= time.monotonic():
            self._count("inference_requests_expired_total")
            raise Expired("deadline already exceeded on arrival")

        fut: "asyncio.Future" = asyncio.get_running_loop().create_future()
        item = _Item(features, fut, version, deadline)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._count("inference_requests_rejected_total")
            raise QueueFull(
                f"inference queue full ({self._queue.maxsize} deep)"
            ) from None
        self._gauge("inference_queue_depth", self._queue.qsize())
        return await fut

    # ---- batching loop ---------------------------------------------------
    async def _loop(self) -> None:
        try:
            while True:
                first = await self._next_item()
                if first is None:
                    return
                batch = await self._fill(first)
                self._run_batch(batch)
        except asyncio.CancelledError:
            raise
        finally:
            self._gauge("inference_queue_depth", self._queue.qsize())

    async def _next_item(self) -> "Optional[_Item]":
        """Block for the first item; return None once stopping and drained."""
        get_task = asyncio.ensure_future(self._queue.get())
        stop_task = asyncio.ensure_future(self._stopping.wait())
        try:
            done, _ = await asyncio.wait(
                {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                return get_task.result()
            # Stopping: serve whatever is still queued, then exit.
            if not self._queue.empty():
                return await get_task
            return None
        finally:
            if not get_task.done():
                get_task.cancel()
            stop_task.cancel()

    async def _fill(self, first: _Item) -> List[_Item]:
        batch = [first]
        deadline = time.monotonic() + self._delay
        while len(batch) < self._max_batch:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        self._adapt(len(batch))
        return batch

    def _adapt(self, filled: int) -> None:
        """Halve the window while batches saturate; ease it back when they don't."""
        if not self._adaptive:
            return
        if filled >= self._max_batch:
            self._delay = max(self._min_delay, self._delay / 2.0)
        elif filled == 1:
            self._delay = min(self._max_delay, max(self._min_delay, self._delay * 2.0))

    def _run_batch(self, batch: List[_Item]) -> None:
        now = time.monotonic()
        live: "Dict[str, List[_Item]]" = {}
        for item in batch:
            if item.future.done():  # client already went away
                continue
            if item.expired(now):
                self._count("inference_requests_expired_total")
                self._fail(item, Expired("deadline exceeded while queued"))
                continue
            live.setdefault(item.version or self._registry.default_version, []).append(item)

        self._gauge("inference_queue_depth", self._queue.qsize())
        for version, items in live.items():
            try:
                model = self._registry.get(version)
                outputs = model.predict_batch([i.features for i in items])
            except Exception as exc:  # model failure must not kill the loop
                self._count("inference_model_errors_total")
                for item in items:
                    self._fail(item, exc)
                continue
            n = len(items)
            self._count("inference_batches_total")
            self._count("inference_batched_requests_total", n)
            self._observe("inference_batch_size", n)
            for item, scores in zip(items, outputs):
                if not item.future.done():
                    item.future.set_result(
                        Result(scores, n, model.version, (now - item.enqueued_at) * 1000.0)
                    )

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _fail(item: _Item, exc: BaseException) -> None:
        if not item.future.done():
            item.future.set_exception(exc)

    def _count(self, name: str, value: float = 1.0) -> None:
        if self._metrics is not None:
            self._metrics.inc(name, value)

    def _gauge(self, name: str, value: float) -> None:
        if self._metrics is not None:
            self._metrics.set_gauge(name, value)

    def _observe(self, name: str, value: float) -> None:
        if self._metrics is not None:
            self._metrics.observe(name, value)

"""Async dynamic batcher — see docs/LLD.md §2."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List

from model import Model


@dataclass
class _Item:
    features: List[float]
    future: "asyncio.Future"


@dataclass
class Result:
    scores: List[float]
    batch_size: int


class DynamicBatcher:
    def __init__(self, model: Model, max_batch: int = 32, max_delay_ms: int = 5) -> None:
        self._model = model
        self._max_batch = max_batch
        self._max_delay = max_delay_ms / 1000.0
        self._queue: "asyncio.Queue[_Item]" = asyncio.Queue()
        self._task: "asyncio.Task | None" = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def submit(self, features: List[float]) -> Result:
        fut: "asyncio.Future" = asyncio.get_running_loop().create_future()
        await self._queue.put(_Item(features, fut))
        return await fut

    async def _loop(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = time.monotonic() + self._max_delay
            while len(batch) < self._max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            outputs = self._model.predict_batch([b.features for b in batch])
            n = len(batch)
            for item, scores in zip(batch, outputs):
                if not item.future.done():
                    item.future.set_result(Result(scores, n))

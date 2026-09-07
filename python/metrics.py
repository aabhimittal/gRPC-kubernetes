"""Zero-dependency metrics: counters, gauges, and a latency histogram.

Exposed in Prometheus text format on a side port so scraping never competes
with gRPC traffic for the serving port, and so a scrape cannot be blocked by a
saturated inference queue.
"""
from __future__ import annotations

import math
import threading
from typing import Dict, List, Tuple

# Buckets in milliseconds — tuned for online inference (sub-ms to ~2s).
DEFAULT_BUCKETS_MS: Tuple[float, ...] = (
    0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500,
)


class Histogram:
    """Cumulative histogram plus a bounded reservoir for exact percentiles."""

    def __init__(self, buckets: Tuple[float, ...] = DEFAULT_BUCKETS_MS,
                 reservoir: int = 4096) -> None:
        self._buckets = buckets
        self._counts = [0] * (len(buckets) + 1)  # last cell = +Inf
        self._sum = 0.0
        self._n = 0
        self._ring: List[float] = []
        self._ring_cap = reservoir
        self._ring_pos = 0
        self._lock = threading.Lock()

    def observe(self, value_ms: float) -> None:
        with self._lock:
            self._sum += value_ms
            self._n += 1
            idx = len(self._buckets)
            for i, b in enumerate(self._buckets):
                if value_ms <= b:
                    idx = i
                    break
            self._counts[idx] += 1
            if len(self._ring) < self._ring_cap:
                self._ring.append(value_ms)
            else:
                self._ring[self._ring_pos] = value_ms
                self._ring_pos = (self._ring_pos + 1) % self._ring_cap

    def percentile(self, q: float) -> float:
        """Nearest-rank percentile over the reservoir. q in [0, 1]."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        with self._lock:
            if not self._ring:
                return 0.0
            ordered = sorted(self._ring)
            rank = max(1, math.ceil(q * len(ordered))) if q > 0 else 1
            return ordered[rank - 1]

    def snapshot(self) -> Tuple[List[int], float, int]:
        with self._lock:
            return list(self._counts), self._sum, self._n


class Registry:
    """Counters + gauges + histograms, rendered in Prometheus text format."""

    def __init__(self) -> None:
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._hists: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def gauge(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, 0.0)

    def histogram(self, name: str) -> Histogram:
        with self._lock:
            h = self._hists.get(name)
            if h is None:
                h = Histogram()
                self._hists[name] = h
            return h

    def observe(self, name: str, value_ms: float) -> None:
        self.histogram(name).observe(value_ms)

    def render(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        lines: List[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            hists = dict(self._hists)
        for name, v in sorted(counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {v:g}")
        for name, v in sorted(gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {v:g}")
        for name, h in sorted(hists.items()):
            counts, total, n = h.snapshot()
            lines.append(f"# TYPE {name} histogram")
            cum = 0
            for i, b in enumerate(h._buckets):
                cum += counts[i]
                lines.append(f'{name}_bucket{{le="{b:g}"}} {cum}')
            cum += counts[-1]
            lines.append(f'{name}_bucket{{le="+Inf"}} {cum}')
            lines.append(f"{name}_sum {total:g}")
            lines.append(f"{name}_count {n}")
        return "\n".join(lines) + "\n"


async def serve_metrics(registry: Registry, port: int):
    """Minimal asyncio HTTP server exposing GET /metrics and GET /healthz.

    Deliberately dependency-free: pulling in a web framework just to expose
    counters would bloat the inference image and its attack surface.
    """
    import asyncio

    async def handle(reader: "asyncio.StreamReader", writer: "asyncio.StreamWriter") -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            path = request_line.decode("latin-1").split(" ")[1] if b" " in request_line else "/"
            if path.startswith("/metrics"):
                body = registry.render().encode()
                ctype = "text/plain; version=0.0.4"
                status = "200 OK"
            elif path.startswith("/healthz"):
                body, ctype, status = b"ok\n", "text/plain", "200 OK"
            else:
                body, ctype, status = b"not found\n", "text/plain", "404 Not Found"
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, "0.0.0.0", port)

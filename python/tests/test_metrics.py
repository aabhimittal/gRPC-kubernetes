"""Metrics must stay correct at the edges — an SLO dashboard built on a broken
percentile is worse than no dashboard."""
import asyncio

import pytest

from metrics import DEFAULT_BUCKETS_MS, Histogram, Registry, serve_metrics


def test_empty_histogram_percentile_is_zero_not_an_error():
    assert Histogram().percentile(0.99) == 0.0


def test_percentiles_are_nearest_rank():
    h = Histogram()
    for v in range(1, 101):          # 1..100 ms
        h.observe(float(v))
    assert h.percentile(0.5) == 50.0
    assert h.percentile(0.95) == 95.0
    assert h.percentile(0.99) == 99.0
    assert h.percentile(1.0) == 100.0
    assert h.percentile(0.0) == 1.0


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_percentile_rejects_out_of_range_quantiles(q):
    with pytest.raises(ValueError):
        Histogram().percentile(q)


def test_reservoir_is_bounded_and_keeps_recent_samples():
    """Long-running pods must not grow a sample buffer forever."""
    h = Histogram(reservoir=8)
    for v in range(1000):
        h.observe(float(v))
    assert len(h._ring) == 8
    assert h.percentile(1.0) == 999.0        # newest sample retained
    _, total, n = h.snapshot()
    assert n == 1000                          # counts stay exact regardless
    assert total == sum(float(v) for v in range(1000))


def test_bucket_counts_are_cumulative_and_include_overflow():
    h = Histogram()
    h.observe(0.1)                            # first bucket
    h.observe(10_000.0)                       # beyond the largest bucket -> +Inf
    r = Registry()
    r._hists["latency_ms"] = h
    text = r.render()
    assert 'latency_ms_bucket{le="0.5"} 1' in text
    assert f'latency_ms_bucket{{le="{DEFAULT_BUCKETS_MS[-1]:g}"}} 1' in text
    assert 'latency_ms_bucket{le="+Inf"} 2' in text
    assert "latency_ms_count 2" in text


def test_boundary_value_lands_in_its_own_bucket():
    h = Histogram()
    h.observe(5.0)                            # exactly a bucket edge: le is inclusive
    counts, _, _ = h.snapshot()
    assert counts[DEFAULT_BUCKETS_MS.index(5)] == 1


def test_counters_and_gauges_render_in_prometheus_format():
    r = Registry()
    r.inc("requests_total")
    r.inc("requests_total", 4)
    r.set_gauge("queue_depth", 7)
    r.set_gauge("queue_depth", 3)             # gauges replace, counters add
    text = r.render()
    assert "requests_total 5" in text
    assert "queue_depth 3" in text
    assert "# TYPE requests_total counter" in text
    assert "# TYPE queue_depth gauge" in text
    assert text.endswith("\n")


def test_unknown_counter_reads_as_zero():
    assert Registry().counter("never_touched") == 0.0


def test_metrics_endpoint_serves_scrapes_and_404s_everything_else():
    async def body():
        registry = Registry()
        registry.inc("inference_requests_total", 3)
        server = await serve_metrics(registry, 0)
        port = server.sockets[0].getsockname()[1]

        async def get(path):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            await writer.drain()
            data = await reader.read()
            writer.close()
            return data.decode()

        assert "inference_requests_total 3" in await get("/metrics")
        assert "200 OK" in await get("/healthz")
        assert "404" in await get("/nope")
        server.close()

    asyncio.run(body())

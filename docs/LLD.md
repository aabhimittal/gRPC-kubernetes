# Low-Level Design — Implementation Details

Cross-reference: HLD.md for the big picture. This doc is the "how".

## 1. Contract (`proto/inference.proto`)

Three RPCs: `Predict` (unary), `PredictStream` (bi-di), `HealthCheck`. Every
response carries `server_latency_ms` and `batch_size` so clients can observe
the batching optimization directly. Stubs are generated at build time:

- Python: `grpcio-tools` → `python/gen/*_pb2*.py`
- Go: `protoc-gen-go` + `protoc-gen-go-grpc` → `go/gen/inferencev1/*.go`

## 2. Dynamic batching (core optimization)

Both servers implement the same algorithm; only the concurrency primitive
differs.

```
enqueue(req):
    put (req, promise) on channel
    return promise            # caller awaits its own result

batch_loop():                 # single consumer goroutine / asyncio task
    loop:
        first = blocking_get(queue)          # wait for at least one
        batch = [first]
        deadline = now + max_delay_ms
        while len(batch) < max_batch and now < deadline:
            item = get(queue, timeout=deadline-now)   # non-blocking-ish
            if item: batch.append(item)
        outputs = model.predict_batch([b.req.features for b in batch])
        for b, out in zip(batch, outputs):
            b.promise.set(out, batch_size=len(batch))
```

Tunables (env / ConfigMap): `MAX_BATCH` (default 32), `MAX_DELAY_MS`
(default 5). Small delay trades a few ms of tail latency for large throughput
gains; at low load the deadline fires with batch size 1 so latency is
unaffected.

**Why a single consumer:** serializing model calls avoids accelerator
contention; parallelism comes from *batch width*, not concurrent model calls.

## 3. Model

A deterministic stand-in (`softmax(W·x)`) with a fixed seeded weight matrix so
Python and Go produce comparable outputs without shipping a real checkpoint.
`predict_batch` accepts a 2-D array so a real backend (ONNX Runtime, PyTorch,
Triton) drops in behind the same interface. Warm-up runs `predict_batch` on a
dummy batch at startup.

## 4. Server concurrency

- **Python:** `grpc.aio` async server. The batch loop is an `asyncio.Task`;
  each RPC awaits a per-request `asyncio.Future`. Async server means we don't
  burn threads while requests sit in the batch window.
- **Go:** standard `grpc.Server`. Batcher uses a buffered channel + one
  goroutine; each RPC blocks on a per-request response channel.

Both bound in-flight work: Go via channel capacity, Python via the batch
window naturally serializing.

## 5. gRPC on Kubernetes routing

A plain ClusterIP Service load-balances TCP connections, but gRPC keeps one
long-lived HTTP/2 connection, so all requests pin to one pod. Fixes:

- **Client-side LB (used here):** headless Service + `round_robin` gRPC config
  so the client spreads streams across pod IPs.
- **Proxy LB (production):** an L7 proxy (Envoy / Linkerd / an ingress with
  gRPC support) that balances per-request.

`k8s/service.yaml` ships both a normal and a headless Service; clients use the
headless one with `dns:///` + round-robin.

## 6. Health & lifecycle

- `HealthCheck` returns `SERVING` only after warm-up completes → wired to the
  **readiness** probe so traffic waits for a warm model.
- Liveness uses the same RPC; a hung batch loop fails it and the pod restarts.
- `terminationGracePeriodSeconds` + graceful `Stop()` drain in-flight batches.

## 7. Config surface (ConfigMap `inference-config`)

| Key | Default | Meaning |
| --- | --- | --- |
| `MAX_BATCH` | 32 | max requests per model call |
| `MAX_DELAY_MS` | 5 | max time to fill a batch |
| `MODEL_VERSION` | v1 | reported in responses |
| `PORT` | 50051 | gRPC listen port |

## 8. Observability hooks

`server_latency_ms` and `batch_size` on every response are the minimum viable
signal. Production would add Prometheus counters (batch size histogram, queue
depth) and export queue depth as the HPA custom metric — noted in HLD §4.6.

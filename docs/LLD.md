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

### 2.1 Admission control

The queue is bounded (`MAX_QUEUE`). Past its depth the server returns
`RESOURCE_EXHAUSTED` immediately instead of accepting work it cannot finish in
time. An unbounded queue does not add capacity — it converts overload into a
latency tail that outlives every client deadline, so the work is done *and*
thrown away. Size it so worst-case wait (`MAX_QUEUE / throughput`) stays inside
the client's deadline.

### 2.2 Deadline-aware batching

Every request carries the client's remaining deadline into the queue. Items
whose deadline has already passed are failed with `DEADLINE_EXCEEDED` *before*
the model call and never counted in the batch. Under a latency spike this is
what stops the server spending accelerator time on answers nobody is waiting
for — the difference between a queue that recovers and one that collapses.

### 2.3 Adaptive batch window

The wait halves (down to `MIN_DELAY_US`) whenever a window fills to
`MAX_BATCH` — at that arrival rate waiting buys nothing — and doubles back
toward `MAX_DELAY_MS` when a window closes with a single request. A quiet
service therefore stops paying a fixed per-request tax, and a busy one stops
adding avoidable delay. Set `ADAPTIVE_BATCHING=0` for a fixed window.

### 2.4 Version-aware grouping

Requests pinned to different `model_version`s inside one window are split into
one model call per version. Canary traffic neither blocks the incumbent's batch
nor gets silently served by the wrong weights.

### 2.5 Draining shutdown

On SIGTERM the loop serves what is already queued, then fails anything still
waiting with `UNAVAILABLE` — a retryable code — rather than leaving callers to
discover the shutdown via their own timeouts.

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
  **readiness** probe so traffic waits for a warm model. Every resident model
  version is warmed, not just the default: a cold canary reads as a latency
  regression.
- A **startup probe** covers warm-up so an unusually slow cold start is not
  mistaken for a hung process and restarted.
- Liveness uses the same RPC; a hung batch loop fails it and the pod restarts.
- Shutdown order is deliberate, because endpoint removal is asynchronous:
  1. flip health to `NOT_SERVING` (readiness fails, kube-proxy and client-side
     load balancers start removing this pod),
  2. keep serving for `PRESTOP_SLEEP_MS` while that propagates,
  3. `GracefulStop` in-flight RPCs within `GRACE_MS`, forcing a stop after,
  4. drain the batcher: queued work is served or failed with `UNAVAILABLE`.
  `terminationGracePeriodSeconds` must exceed `PRESTOP_SLEEP_MS + GRACE_MS`, or
  the kubelet SIGKILLs the pod mid-drain — `scripts/validate_manifests.py`
  enforces this.

## 7. Config surface (ConfigMap `inference-config`)

| Key | Default | Meaning |
| --- | --- | --- |
| `MAX_BATCH` | 32 | max requests per model call |
| `MAX_DELAY_MS` | 5 | upper bound on the batch window |
| `MIN_DELAY_US` | 200 | floor the adaptive controller may shrink to |
| `ADAPTIVE_BATCHING` | 1 | `0` pins the window to `MAX_DELAY_MS` |
| `MAX_QUEUE` | 1024 | queue depth before shedding (`RESOURCE_EXHAUSTED`) |
| `STREAM_CONCURRENCY` | 32 | in-flight requests per bidi stream |
| `MODEL_VERSIONS` | v1 | resident versions, `name[:seed]`, comma separated |
| `MODEL_VERSION` | v1 | default version when the request pins nothing |
| `PORT` | 50051 | gRPC listen port |
| `METRICS_PORT` | 9090 | Prometheus / `healthz` side port |
| `KEEPALIVE_MS` | 30000 | HTTP/2 keepalive ping interval |
| `MAX_CONN_AGE_MS` | 300000 | connection recycling, so scale-out rebalances |
| `MAX_MSG_BYTES` | 4194304 | per-message size cap |
| `MAX_CONCURRENT_STREAMS` | 1000 | per-connection stream cap (Go) |
| `PRESTOP_SLEEP_MS` | 3000 | keep serving after readiness fails |
| `GRACE_MS` | 10000 | graceful-stop budget for in-flight RPCs |

## 8. Observability

`server_latency_ms` and `batch_size` on every response are the per-request
signal. Fleet-level signal is exposed in Prometheus text format on
`METRICS_PORT` (`/metrics`, plus `/healthz`) — a side port, so a scrape is
never queued behind saturated inference traffic and never competes for the
gRPC port.

| Metric | Type | Reads as |
| --- | --- | --- |
| `inference_requests_total` / `inference_responses_total` | counter | unary traffic in / out |
| `inference_stream_requests_total` / `..._responses_total` | counter | streaming traffic |
| `inference_batches_total` / `inference_batched_requests_total` | counter | ratio = mean batch size |
| `inference_batch_size` | histogram | is batching actually engaging? |
| `inference_latency_ms` | histogram | server-side latency distribution |
| `inference_queue_depth` | gauge | the HPA signal (leads CPU) |
| `inference_requests_rejected_total` | counter | load shed by admission control |
| `inference_requests_expired_total` | counter | work dropped on a dead deadline |
| `inference_invalid_total` / `inference_errors_total` | counter | bad input vs. server fault |

Queue depth leads CPU by roughly one batch window, which makes it the better
HPA metric — `k8s/hpa.yaml` carries the custom-metric block, commented, next to
the CPU target that works without a metrics adapter.

## 9. Failure modes and the codes they map to

The status code is the contract: it tells the caller whether to fix the
request, retry here, or retry elsewhere.

| Condition | Code | Client action |
| --- | --- | --- |
| empty / wrong-length / NaN / infinite features | `INVALID_ARGUMENT` | fix the request; never retry |
| vector over `MaxFeatures` (4096) | `INVALID_ARGUMENT` | fix the request |
| `model_version` not resident | `NOT_FOUND` | fix the pin (the message lists what is loaded) |
| queue full | `RESOURCE_EXHAUSTED` | back off, retry — safe, no model ran |
| deadline passed while queued | `DEADLINE_EXCEEDED` | budget more time or shed |
| server draining | `UNAVAILABLE` | retry another pod — safe, no model ran |
| model raised | `INTERNAL` | the batch fails, the loop survives |

Both demo clients configure a gRPC retry policy for `UNAVAILABLE` and
`RESOURCE_EXHAUSTED` only: those are returned *before* the model runs, so a
retry cannot duplicate work.

## 10. Test surface

`make test` runs both suites; CI runs them on every push.

- **Unit** — batching windows, admission control, deadline eviction, adaptive
  clamping, version routing, model faults, shutdown drain, metric arithmetic,
  input validation (`python/tests/`, `go/internal/*/`_test.go`).
- **End-to-end over real gRPC** — status-code mapping, canary pinning, stream
  pipelining, load shedding, readiness transitions (`python/tests/
  test_server_e2e.py`, `go/cmd/server/main_test.go`, on bufconn).
- **Cross-language parity** — `testdata/parity.json` holds golden vectors both
  suites assert against, so the two backends behind one Service cannot drift.
- **Manifests** — `scripts/validate_manifests.py` checks probe/port agreement
  with the ConfigMap, grace period vs. actual drain time, and rollout
  capacity.
- Go tests run under `-race`.

# High-Level Design — gRPC Inference on Kubernetes

## 1. Problem

Serve a low-latency, high-throughput ML inference API. Naive "one HTTP
request → one model call" wastes the two most expensive resources in an
inference stack: the accelerator (idle between single requests) and the
network (a fresh connection + serialization per call). This project shows how
gRPC + Kubernetes remove both bottlenecks.

## 2. Why gRPC

| Concern | REST/JSON | gRPC |
| --- | --- | --- |
| Wire format | text JSON | Protobuf binary (smaller, faster to parse) |
| Transport | HTTP/1.1, 1 req/conn | HTTP/2, many multiplexed streams/conn |
| Streaming | bolt-on (SSE/WS) | first-class bi-di streaming |
| Contract | OpenAPI (optional) | `.proto` is the enforced contract |

For inference the streaming + multiplexing matter most: one warm connection
carries thousands of predictions, and bi-di streaming lets a client fire
inputs without waiting for each response.

## 3. Architecture

```
                 ┌──────────────────────── Kubernetes cluster ───────────────────────┐
                 │                                                                    │
  client ──gRPC──┼──▶ Service (ClusterIP, HTTP/2) ──▶ Deployment (N pods, HPA-scaled) │
   (Py/Go)       │                                     │                              │
                 │                                     ▼                              │
                 │                            ┌──────────────────┐                    │
                 │                            │  inference pod   │                    │
                 │                            │  ┌────────────┐  │                    │
                 │                            │  │  Batcher   │  │  dynamic batching  │
                 │                            │  └─────┬──────┘  │                    │
                 │                            │        ▼         │                    │
                 │                            │  ┌────────────┐  │                    │
                 │                            │  │   Model    │  │  cached / warmed   │
                 │                            │  └────────────┘  │                    │
                 │                            └──────────────────┘                    │
                 └────────────────────────────────────────────────────────────────────┘
```

## 4. Optimization strategies (the "novel approaches")

1. **Dynamic (adaptive) batching.** The server holds incoming requests for at
   most `max_delay_ms` OR until `max_batch` accumulate, whichever comes first,
   then runs them as one model call. Turns many tiny accelerator calls into
   few large ones — the single biggest throughput lever. See LLD §2.
2. **HTTP/2 multiplexing + long-lived channels.** Clients keep one channel and
   send concurrent RPCs over it, eliminating per-request TCP/TLS setup.
3. **Bi-directional streaming.** `PredictStream` amortizes batching + framing
   across a whole sequence; ideal for feature pipelines and re-ranking.
4. **Model warm-up + caching.** The model is loaded and exercised once at pod
   start so the first real request doesn't pay cold-start latency; readiness
   gate holds traffic until warm.
5. **Right-sized concurrency.** A bounded worker pool prevents accelerator
   thrash; backpressure is expressed through gRPC flow control, not unbounded
   queues.
6. **Horizontal autoscaling.** HPA scales pods on CPU (and can be extended to
   custom queue-depth / latency metrics) so batching handles per-pod
   efficiency while HPA handles fleet-level load. Scaling *behavior* is tuned
   for warm-up cost: up fast, down slowly.

### 4.7 Degrading well

Throughput optimizations decide how the service performs when it is healthy;
these decide what happens when it is not. They are implemented in both
backends (LLD §2.1–2.5, §9).

- **Shed, don't queue.** Capacity is finite; a queue past that point converts
  overload into a latency tail and produces answers no client is still waiting
  for. The server bounds its queue and returns `RESOURCE_EXHAUSTED` — a
  retryable code the caller's load balancer can act on.
- **Respect the client's clock.** Deadlines propagate into the batcher, and
  expired work is dropped before the model call. Spending accelerator time on
  a dead request is how a recoverable spike turns into a collapse.
- **Fail with the right code.** `INVALID_ARGUMENT` (fix the request),
  `NOT_FOUND` (fix the pin), `RESOURCE_EXHAUSTED` / `UNAVAILABLE` (retry — no
  model ran, so a retry is safe), `DEADLINE_EXCEEDED` (budget more time). The
  code is the contract; retry policies are built on it.
- **Roll out without a redeploy.** Several model versions stay resident and
  warm; `model_version` on the request routes to one, so a canary is a routing
  decision and a rollback is instant.
- **Drain, don't drop.** Readiness fails first, the pod keeps serving while
  endpoint removal propagates, then in-flight work finishes and the remainder
  fails retryably.

## 5. Kubernetes shape

- **Deployment** per language (`python`, `go`) — identical proto, so a client
  can hit either.
- **Service** (ClusterIP) fronts each Deployment; gRPC needs HTTP/2, so an
  L7-aware ingress or headless+client-side LB is used for real routing
  (documented in LLD §5).
- **HPA** targets 60% CPU, 2–10 replicas, with `behavior` tuned so bursts scale
  up quickly and scale-down is deliberate (warm pods are expensive to discard).
- **Probes** call `HealthCheck` via `grpc_health_probe`-style exec / native
  gRPC probes: a startup probe covers warm-up, readiness gates traffic, and
  liveness catches a wedged batch loop.
- **PodDisruptionBudget** keeps at least one replica through node drains and
  cluster upgrades; topology spread keeps replicas off a single node.
- **Metrics** are scraped from a side port (`/metrics`), never the serving
  port, so observability survives saturation.

## 6. Non-goals

Real model training, GPU scheduling specifics, and auth/mTLS are out of scope
for the demo but flagged where they'd plug in.

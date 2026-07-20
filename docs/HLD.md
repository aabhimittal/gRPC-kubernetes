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
   efficiency while HPA handles fleet-level load.

## 5. Kubernetes shape

- **Deployment** per language (`python`, `go`) — identical proto, so a client
  can hit either.
- **Service** (ClusterIP) fronts each Deployment; gRPC needs HTTP/2, so an
  L7-aware ingress or headless+client-side LB is used for real routing
  (documented in LLD §5).
- **HPA** targets 60% CPU, 2–10 replicas.
- **Probes** call `HealthCheck` via `grpc_health_probe`-style exec / native
  gRPC probes.

## 6. Non-goals

Real model training, GPU scheduling specifics, and auth/mTLS are out of scope
for the demo but flagged where they'd plug in.

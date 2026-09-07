# gRPC Inference on Kubernetes

A working demo of low-latency, high-throughput ML **inference over gRPC**,
deployed on **Kubernetes**, with the inference-optimization techniques from the
HLD/LLD design docs. The same `.proto` is implemented **twice — Python and Go —**
so a client can hit either backend.

> Design write-ups: [`docs/HLD.md`](docs/HLD.md) (architecture & strategy),
> [`docs/LLD.md`](docs/LLD.md) (implementation detail).

## What it demonstrates

| Optimization | Where |
| --- | --- |
| **Dynamic batching** — coalesce concurrent requests into one model call | `python/batcher.py`, `go/internal/batch/batcher.go` |
| **HTTP/2 multiplexing + long-lived channels** | clients in `python/client.py`, `go/cmd/client` |
| **Bi-directional streaming** (`PredictStream`) | servers + clients |
| **Model warm-up gated by readiness** | both servers, wired to gRPC health probe |
| **Client-side round-robin LB** over a headless Service | `k8s/service.yaml` + clients |
| **Horizontal autoscaling** (HPA on CPU) | `k8s/hpa.yaml` |

Every `PredictResponse` reports `server_latency_ms` and `batch_size`, so you can
*see* batching engage under load.

## Behaviour under stress

The optimizations above are the easy half. These are the controls that decide
what happens when traffic stops being polite — each implemented in **both**
backends and covered by tests.

| Control | What it prevents | Where |
| --- | --- | --- |
| **Admission control** — bounded queue, sheds with `RESOURCE_EXHAUSTED` | overload silently becoming an unbounded latency tail | `MAX_QUEUE` |
| **Deadline-aware batching** — expired work dropped before the model call | burning accelerator time on answers nobody is waiting for | client deadline → batcher |
| **Adaptive batch window** — shrinks under load, recovers when idle | a quiet service paying a fixed per-request delay tax | `MIN_DELAY_US`, `ADAPTIVE_BATCHING` |
| **Multi-version registry + `model_version` pinning** | a canary rollout requiring a redeploy, or blending into the incumbent's batch | `MODEL_VERSIONS` |
| **Input validation** with precise status codes | one bad client (NaN, truncated vector, 10 MB payload) degrading everyone | `python/validation.py`, `go/internal/validate` |
| **Pipelined streaming** — many in-flight per stream | one slow item stalling a stream, and single-client streams never batching | `STREAM_CONCURRENCY` |
| **Draining shutdown** — readiness fails first, then drain | rollouts showing up as client-visible errors | `PRESTOP_SLEEP_MS`, `GRACE_MS` |
| **Prometheus metrics on a side port** | scrapes queueing behind saturated inference traffic | `METRICS_PORT` → `/metrics` |
| **Client retry policy** on pre-model failures only | retries duplicating model work | both clients' service config |

Failure modes and the gRPC code each maps to are tabulated in
[`docs/LLD.md` §9](docs/LLD.md); the metric surface is §8.

```bash
# Watch the server shed load rather than melt (queue of 4, batch of 2):
MAX_QUEUE=4 MAX_BATCH=2 MAX_DELAY_MS=20 python python/server.py &
python python/client.py --mode load --n 2000 --concurrency 400
# The client tallies status codes instead of dying on the first failure, so the
# run reports what got shed (RESOURCE_EXHAUSTED), what timed out, and the
# client-side p50/p95/p99 — its retry policy absorbs some of the shedding,
# which is the point of returning a retryable code.

# Canary a second version without a redeploy:
MODEL_VERSIONS=v1,v2-canary:987654 python python/server.py &
python python/client.py --mode unary --version v2-canary

curl localhost:9090/metrics | grep inference_
```

## Tests

```bash
make test        # both suites
make test-py     # pytest: unit + end-to-end gRPC
make test-go     # go test -race: unit + end-to-end gRPC (bufconn)
make lint        # gofmt, go vet, manifest checks
```

What is covered, beyond the happy path: overload shedding, client timeouts
expiring in the queue, cancelled callers, malformed and adversarial inputs,
unknown version pins, model faults that must not kill the batch loop, shutdown
draining, adaptive-window clamping, metric arithmetic at the edges, and
**cross-language parity** (`testdata/parity.json` — both backends serve one
Service, so a client must not be able to tell which answered).

## Layout

```
proto/inference.proto        # the contract (source of truth)
python/                      # asyncio grpc.aio server + client
  model.py  batcher.py       #   model registry, dynamic batcher
  validation.py  metrics.py  #   input guards, Prometheus registry
  tests/                     #   unit + end-to-end gRPC tests
go/                          # standard grpc server + client
  internal/inference|batch/  #   model registry, dynamic batcher
  internal/validate|metrics/ #   input guards, Prometheus registry
  cmd/server/main_test.go    #   end-to-end gRPC tests over bufconn
k8s/                         # namespace, configmap, deployments, services,
                             #   HPA (with behavior), PodDisruptionBudget
scripts/                     # parity fixture generator, manifest validator
testdata/parity.json         # golden vectors both languages must agree on
docs/                        # HLD.md, LLD.md
.github/workflows/ci.yml     # go (race) + python + manifest jobs
Makefile                     # gen / run / test / lint / docker / deploy
```

## Quick start (local)

Prereqs: `python3` + `pip`, `go` ≥ 1.24, and `protoc` (+ Go plugins) for Go
codegen. Stubs are generated, not committed.

```bash
# Python
make py-run                                   # generates stubs, starts server on :50051
python python/client.py --mode load --n 500 --concurrency 50

# Go (separate terminal / different port via PORT=)
make go-run
go run ./go/cmd/client -mode load -n 500 -concurrency 50
```

Under concurrency the client prints `avg server batch_size > 1` — that's the
dynamic batcher turning many RPCs into few model calls.

## Docker

```bash
make docker        # builds inference-py:latest and inference-go:latest
```

Both Dockerfiles use the **repo root** as build context (they need `proto/`) and
run codegen inside the build.

## Kubernetes

```bash
make k8s-deploy                     # namespace + configmap + deployments + services + HPA
kubectl -n inference get pods,svc,hpa

# Try it from inside the cluster:
kubectl -n inference run c --rm -it --image=inference-go:latest --restart=Never -- \
  /server --help    # or exec a client against inference-go-headless:50051
```

Every tunable lives in `k8s/configmap.yaml` — batching, admission control,
model versions, transport keepalive, and shutdown timing — documented key by
key in [`docs/LLD.md` §7](docs/LLD.md). The manifests also carry a
PodDisruptionBudget, topology spread, a startup probe for warm-up, non-root
read-only containers, and HPA `behavior` tuned for pods that pay a warm-up cost
(scale up fast, scale down slowly).

`scripts/validate_manifests.py` (run by `make lint` and CI) checks the parts
that only fail at rollout time: probes pointing at the declared gRPC port, the
metrics port matching the ConfigMap, `terminationGracePeriodSeconds` actually
covering the server's own drain, and rollouts that never dip below capacity.

## Model note

The bundled model is a deterministic `softmax(W·x)` stand-in with seeded weights
(identical in Python and Go) so the demo runs with no checkpoint. It sits behind
a `predict_batch(matrix) -> matrix` interface — swap in ONNX Runtime, PyTorch, or
Triton without touching the server or batcher.

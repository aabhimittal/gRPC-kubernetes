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

## Layout

```
proto/inference.proto        # the contract (source of truth)
python/                      # asyncio grpc.aio server + client
go/                          # standard grpc server + client
k8s/                         # namespace, configmap, deployments, services, HPA
docs/                        # HLD.md, LLD.md
Makefile                     # gen / run / docker / deploy targets
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

Batching tunables (`MAX_BATCH`, `MAX_DELAY_MS`, `MODEL_VERSION`) live in
`k8s/configmap.yaml`.

## Model note

The bundled model is a deterministic `softmax(W·x)` stand-in with seeded weights
(identical in Python and Go) so the demo runs with no checkpoint. It sits behind
a `predict_batch(matrix) -> matrix` interface — swap in ONNX Runtime, PyTorch, or
Triton without touching the server or batcher.

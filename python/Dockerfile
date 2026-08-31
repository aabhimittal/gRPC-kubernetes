# Build context = repo root:  docker build -f python/Dockerfile -t inference-py .
FROM python:3.11-slim AS build
WORKDIR /app
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY python/ .
COPY proto/ ./proto/
# Generate stubs into gen/ and patch the grpc stub's import to be relative.
RUN python -m grpc_tools.protoc -I./proto \
      --python_out=gen --grpc_python_out=gen proto/inference.proto \
 && sed -i 's/^import inference_pb2/from . import inference_pb2/' gen/inference_pb2_grpc.py

FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir grpcio==1.68.1 grpcio-health-checking==1.68.1
COPY --from=build /app /app
# Run unprivileged with a read-only root filesystem (see k8s securityContext);
# PYTHONDONTWRITEBYTECODE keeps the interpreter from trying to write .pyc.
RUN useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin nonroot
USER 65532:65532
ENV PORT=50051 METRICS_PORT=9090 \
    MAX_BATCH=32 MAX_DELAY_MS=5 MIN_DELAY_US=200 ADAPTIVE_BATCHING=1 \
    MAX_QUEUE=1024 STREAM_CONCURRENCY=32 \
    MODEL_VERSIONS=v1 MODEL_VERSION=v1 \
    PRESTOP_SLEEP_MS=3000 GRACE_MS=10000 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 50051 9090
CMD ["python", "server.py"]

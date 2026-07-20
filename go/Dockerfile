# Build context = repo root:  docker build -f go/Dockerfile -t inference-go .
FROM golang:1.24 AS build
WORKDIR /src

# protoc + Go plugins for codegen.
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2 \
 && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1
ENV PATH="/go/bin:${PATH}"

COPY go/go.mod go/go.sum* ./go/
WORKDIR /src/go
RUN go mod download || true

WORKDIR /src
COPY go/ ./go/
COPY proto/ ./proto/
# Generate stubs into go/gen (paths controlled by go_package option in the proto).
RUN protoc -I./proto \
      --go_out=go/gen --go_opt=module=github.com/aabhimittal/grpc-kubernetes/go/gen \
      --go-grpc_out=go/gen --go-grpc_opt=module=github.com/aabhimittal/grpc-kubernetes/go/gen \
      proto/inference.proto

WORKDIR /src/go
RUN go mod tidy && CGO_ENABLED=0 go build -o /out/server ./cmd/server

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/server /server
ENV PORT=50051 MAX_BATCH=32 MAX_DELAY_MS=5 MODEL_VERSION=v1
EXPOSE 50051
ENTRYPOINT ["/server"]

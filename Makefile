.PHONY: help gen gen-python gen-go py-run go-run test test-py test-go lint parity docker k8s-deploy k8s-delete clean

IMG_PY ?= inference-py:latest
IMG_GO ?= inference-go:latest
GO_MODULE := github.com/aabhimittal/grpc-kubernetes/go/gen

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

gen: gen-python gen-go ## Generate all stubs

gen-python: ## Generate Python stubs into python/gen
	cd python && python -m grpc_tools.protoc -I../proto \
	  --python_out=gen --grpc_python_out=gen ../proto/inference.proto
	# Patch absolute import -> package-relative so `from gen import ...` works.
	sed -i.bak 's/^import inference_pb2/from . import inference_pb2/' python/gen/inference_pb2_grpc.py && rm -f python/gen/inference_pb2_grpc.py.bak

gen-go: ## Generate Go stubs into go/gen (needs protoc + plugins on PATH)
	protoc -I./proto \
	  --go_out=go/gen --go_opt=module=$(GO_MODULE) \
	  --go-grpc_out=go/gen --go-grpc_opt=module=$(GO_MODULE) \
	  proto/inference.proto
	cd go && go mod tidy

py-run: gen-python ## Run the Python server locally
	cd python && python server.py

go-run: gen-go ## Run the Go server locally
	cd go && go run ./cmd/server

test: test-py test-go ## Run every test suite

test-py: gen-python ## Python unit + end-to-end gRPC tests
	python -m pytest python/tests -q

test-go: ## Go unit + end-to-end gRPC tests (race detector on)
	cd go && go test -race ./...

lint: ## Formatting / vet / manifest checks that CI enforces
	cd go && gofmt -l . && go vet ./...
	python scripts/validate_manifests.py

parity: ## Regenerate the cross-language golden vectors in testdata/
	python scripts/gen_parity.py

docker: ## Build both images (build context = repo root)
	docker build -f python/Dockerfile -t $(IMG_PY) .
	docker build -f go/Dockerfile     -t $(IMG_GO) .

k8s-deploy: ## Apply all manifests
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/

k8s-delete: ## Tear down
	kubectl delete -f k8s/ --ignore-not-found

clean:
	rm -f python/gen/inference_pb2*.py
	rm -rf go/gen/inferencev1

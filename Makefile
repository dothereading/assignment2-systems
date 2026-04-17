IMAGE := cs336-systems-cpu

.PHONY: build run test

## Build the CPU-dev Docker image
build:
	docker build -f Dockerfile.cpu-dev -t $(IMAGE) .

## Run an interactive shell with live source edits mounted
run:
	docker run --rm -it \
	  -v "$(CURDIR)/cs336_systems:/workspace/cs336_systems" \
	  -v "$(CURDIR)/tests:/workspace/tests" \
	  $(IMAGE)

## Run the test suite inside the container
test:
	docker run --rm \
	  -v "$(CURDIR)/cs336_systems:/workspace/cs336_systems" \
	  -v "$(CURDIR)/tests:/workspace/tests" \
	  $(IMAGE) \
	  uv run pytest tests/ -v

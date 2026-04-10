# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Local development on macOS (CPU-only Triton)

Triton has no native macOS backend (no Metal, no CUDA).
Use `Dockerfile.cpu-dev` to run a Linux container with
[`triton-cpu`](https://pypi.org/project/triton-cpu/), which compiles
`@triton.jit` kernels via LLVM on CPU — no GPU required.

### Prerequisites

- [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)

### Build the image

```sh
docker build -f Dockerfile.cpu-dev -t cs336-systems-cpu .
```

### Run an interactive shell (with live source edits)

```sh
docker run --rm -it \
  -v "$(pwd)/cs336_systems:/workspace/cs336_systems" \
  -v "$(pwd)/tests:/workspace/tests" \
  cs336-systems-cpu
```

Inside the container you can run Triton kernels on CPU by targeting `device="cpu"` in your kernel launches:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

x = torch.ones(1024, device="cpu")
y = torch.ones(1024, device="cpu")
out = torch.empty_like(x)
add_kernel[(16,)](x, y, out, 1024, BLOCK=64)
```

### Run the tests

```sh
# Inside the container — CUDA-gated Triton tests are skipped automatically;
# pure-PyTorch and CPU-targeted tests run normally.
uv run pytest tests/ -v
```

> **Note:** The CUDA-gated `test_flash_forward_pass_triton` /
> `test_flash_backward_triton` tests will be skipped (`torch.cuda.is_available()`
> returns `False`). Write companion CPU tests using `device="cpu"` to iterate
> on kernel logic locally, then validate on a GPU node before submitting.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

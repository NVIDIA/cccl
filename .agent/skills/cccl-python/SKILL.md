---
name: cccl-python
description: "CCCL's Python packages (`cuda-cccl`): installation, module layout, build/test scripts, test organization. Use when the user works on the Python bindings, builds/tests Python components, or asks about the `cuda.compute` / `cuda.coop` / `cuda.cccl.headers` modules. Trigger phrases: \"cccl python\", \"cuda.compute\", \"cuda.coop\", \"cuda-cccl package\", \"build the python bindings\", \"test python\"."
---

# cccl-python

Python components live under `python/cuda_cccl/`. Build/test scripts take `-py-version` instead of compiler flags.
Supported: Python 3.10 – 3.13.

## Modules

- `cuda.compute` — device-level algorithms, iterators, custom GPU types.
- `cuda.coop._experimental` — block/warp primitives for Numba CUDA.
- `cuda.cccl.headers` — programmatic access to CCCL headers.

## Install from source

```
pip install -e python/cuda_cccl[test-cu13]   # or [test-cu12] for CTK 12.X
```

Requires CTK 12.x or 13.x, NVIDIA GPU CC 6.0+. Base deps: `numba>=0.60.0`, `numpy`, `cuda-pathfinder>=1.2.3`,
`cuda-core`, `typing_extensions`. CUDA extras add `cuda-bindings`, `cuda-toolkit`, `numba-cuda`.

## Build / test

```
./ci/build_cuda_cccl_python.sh        -py-version 3.10
./ci/test_cuda_compute_python.sh      -py-version 3.10
./ci/test_cuda_coop_python.sh         -py-version 3.10
./ci/test_cuda_cccl_headers_python.sh -py-version 3.10
./ci/test_cuda_cccl_examples_python.sh -py-version 3.10
```

Build script needs no GPU; test scripts do.

## Layout

```
python/cuda_cccl/
├── cuda/{compute,coop,cccl/{parallel,cooperative,headers}}/
├── tests/{compute,coop,headers}/  + test_examples.py
├── benchmarks/
└── pyproject.toml
```

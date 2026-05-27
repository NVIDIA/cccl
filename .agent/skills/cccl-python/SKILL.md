---
description: "CCCL's Python package (`cuda-cccl`) under `python/cuda_cccl/`: modules, build/test CI scripts, install extras, layout. Triggers: \"cccl python\", \"cuda.compute\", \"cuda.coop\", \"cuda-cccl package\", \"build the python bindings\", \"test python\"."
---

# cccl-python

Python components live under `python/cuda_cccl/`. Requires Python 3.10+, CTK 12.x or 13.x, GPU CC 6.0+.

## Modules

- `cuda.compute` — device-level algorithms, iterators, custom GPU types.
- `cuda.coop._experimental` — block/warp primitives for Numba CUDA.
- `cuda.cccl.headers` — programmatic access to CCCL headers.

## Install

For dev work from a checkout:

```
pip install -e python/cuda_cccl[test-cu12]   # or [test-cu13] for CTK 13.x
```

PyPI and conda-forge install methods, plus the full requirements list, are in
`references/install.md`.

## Usage examples

```python
import cuda.compute
result = cuda.compute.reduce_into(input_array, output_scalar, init_val, binary_op)

import cuda.coop._experimental as coop
@cuda.jit
def kernel(data):
    coop.block.reduce(data, binary_op)

import cuda.cccl.headers as headers
include_paths = headers.get_include_paths()
```

## Build / test scripts

Scripts under `ci/`; pass `-py-version 3.10` (or 3.11–3.13).

| Script                               | GPU required |
|--------------------------------------|--------------|
| `ci/build_cuda_cccl_python.sh`       | no           |
| `ci/test_cuda_compute_python.sh`     | yes          |
| `ci/test_cuda_coop_python.sh`        | yes          |
| `ci/test_cuda_cccl_headers_python.sh` | yes          |
| `ci/test_cuda_cccl_examples_python.sh` | yes          |

## Layout

```
python/cuda_cccl/
├── cuda/{compute,coop,cccl/{parallel,cooperative,headers}}/
├── tests/{compute,coop,headers}/  + test_examples.py
├── benchmarks/
└── pyproject.toml
```

## Additional resources

- `references/install.md` — PyPI, conda-forge, and source install methods; full requirements.
- `references/docs.md` — index of Python package documentation.
- `references/tools.md` — build and test scripts for the Python bindings.

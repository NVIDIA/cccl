# `cuda.coop`

`cuda.coop` brings CCCL cooperative primitives to Python kernel DSLs. Kernel
code describes a participating thread group and its per-thread values, then
calls the same group-first operations with CUTLASS Python DSL or
Numba-CUDA-MLIR.

```python
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop


@cuda.jit
def exclusive_prefix(values, prefixes):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
    loaded = coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, loaded)
    coop.store(block, prefixes, scanned)
```

The calls inside the kernel are compile-time constructs. The active compiler
lowers them to CCCL providers and accounts for any shared memory they need.

Developers extending the package should start with
[the architecture guide][architecture]. It describes the shared semantic layout,
the intentionally different compiler lifecycles, and the files and tests that
belong to each primitive family.

## Install

The base distribution contains the portable contract and bundled CCCL headers.
Add the extra for the compiler you use:

```console
python -m pip install "cuda-coop[cutlass]"
python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
```

Use `numba-cuda-mlir-cu12` with a CUDA 12 environment. The `cutlass` extra
requires `nvidia-cutlass-dsl>=4.8` and CUDA 13, and cannot resolve from an
index that does not yet carry that DSL release. See [the package
metadata][package-metadata] for the dependency bounds.

## Portable and qualified imports

Use the portable root when a kernel only needs behavior shared by the two
backends:

```python
from cuda import coop
```

Import a qualified module for backend-specific signatures, payload adapters,
callbacks, or optional outputs:

```python
import cuda.coop.cutlass as coop
# or
import cuda.coop.numba_mlir as coop
```

Importing the portable root probes an explicit allowlist containing CUTLASS
Python DSL and Numba-CUDA-MLIR. Each installed candidate must provide the
compiler hooks required by its adapter before `cuda.coop` registers it. A
missing backend is ignored. An incompatible installed backend emits a
`CudaCoopAutoRegistrationWarning` without preventing another compatible
backend from registering.

Set `CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1` before the root import when an
application manages activation itself. A later qualified import validates and
registers that backend:

```python
import os

os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

from cuda import coop
import cuda.coop.cutlass
```

The base import remains usable when neither compiler is installed. Calling a
compiler-dependent operation outside a compatible compilation context reports
a context error.

## Programming model

Every collective receives its participating group as the first argument.
`this_warp()` and `this_block()` refer to physical launch groups.
`ThreadGroup.group_by()` can partition a warp into smaller logical groups or a
block into a group of warps. The operation table below is authoritative:
mapped-block groups are currently backend-qualified, and no portable
collective currently accepts `this_grid()`.

`ThreadData(items_per_thread, dtype=...)` describes a fixed-size register
payload owned by each thread. Most collectives return a fresh payload and leave
their input unchanged. `TempStorage()` asks the compiler to plan scratch space;
`TempStorage(size_in_bytes, alignment=...)` supplies an explicit capacity.
The builtin `int` and `float` dtype tokens mean 32-bit integer and floating-point
values; use an explicit NumPy or compiler dtype token for another width.

The portable surface is:

| Family | Calls | Supported portable groups |
| --- | --- | --- |
| Data movement | `load`, `store`, `exchange` | block, warp, logical warp |
| Reduction | `reduce`, `sum` | thread, block, warp, logical warp, cluster |
| Scan | `scan`, `exclusive_sum`, `inclusive_sum`, `exclusive_scan`, `inclusive_scan` | block, warp, logical warp |
| Neighbors | `adjacent_difference`, `discontinuity`, `shuffle` | block |
| Comparison sort | `merge_sort_keys`, `merge_sort_pairs` | block, warp, logical warp |
| Radix | `radix_sort_keys`, `radix_sort_pairs`, `radix_rank` | block |
| Counting | `histogram`, `run_length_decode` | block |
| Selection | `topk_min_keys`, `topk_min_pairs`, `topk_max_keys`, `topk_max_pairs` | block |

Portable TopK writes an unordered result into the first `k` flattened blocked
positions; the rest of the returned payload is undefined. Operations with a
`valid_items` argument treat it as a uniform prefix length. Consult the shipped
type declarations for the exact dtype, boundary, and result contracts:

- [portable API][portable-stubs]
- [CUTLASS API][cutlass-stubs]
- [Numba-CUDA-MLIR API][numba-stubs]

## CUTLASS provider AOT packs

The CUTLASS backend normally compiles its generated provider bundles as it
traces a workload. An AOT pack records those exact LTO-IR bundles for reuse on
machines with compatible CUDA and `cuda.coop` installations. Capture and
consume packs with the `cuda-coop-aot` command, which requires
`cuda-coop[cutlass]`. Pack capture, inspection, and use currently require
Linux:

```console
cuda-coop-aot capture --output workload.coop-aot -- python workload.py
cuda-coop-aot inspect workload.coop-aot
cuda-coop-aot run --pack workload.coop-aot --mode required -- python workload.py
```

`auto` mode falls back to normal provider compilation after a pack miss,
`required` reports the miss, and `off` ignores the selected pack. The same
controls are available through `cuda.coop.cutlass.aot.capture()` and
`cuda.coop.cutlass.aot.use()`.

Treat an AOT pack as executable device code. Pack digests detect corruption;
they do not authenticate who produced the pack or prove that its LTO-IR is
safe. Consume only packs captured by a build or producer you trust.

Pack reuse matches the provider ABI, exact rendered source, bundle format,
target architecture, compiler options, scratch-layout expressions, and
nvJitLink compatibility. The recorded writer version is informational, and an
exact hit intentionally precedes current-header discovery. A change in shipped
headers that can alter an otherwise identical provider's ABI or semantics must
bump the provider ABI and invalidate older packs.

## Examples

The [examples guide][examples-guide] contains complete programs that
validate their results:

- portable block load, scan, and store with
  [CUTLASS Python DSL][cutlass-scan-example] and
  [Numba-CUDA-MLIR][numba-scan-example]
- qualified radix sort and TopK with
  [CUTLASS Python DSL][cutlass-qualified-example]
- qualified histogram and run-length decode with
  [Numba-CUDA-MLIR][numba-qualified-example]

The package is an alpha API. The
[CCCL documentation](https://nvidia.github.io/cccl/python.html) carries the
release documentation as the public surface stabilizes.

[architecture]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/ARCHITECTURE.md
[package-metadata]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/pyproject.toml
[portable-stubs]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/cuda/coop/__init__.pyi
[cutlass-stubs]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/cuda/coop/cutlass/__init__.pyi
[numba-stubs]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/cuda/coop/numba_mlir/__init__.pyi
[examples-guide]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/examples/README.md
[cutlass-scan-example]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/examples/cutlass/common_block_scan.py
[numba-scan-example]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/examples/numba_mlir/common_block_scan.py
[cutlass-qualified-example]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/examples/cutlass/qualified_radix_topk.py
[numba-qualified-example]: https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop/examples/numba_mlir/qualified_histogram_decode.py

# `cuda.coop`

`cuda.coop` provides portable cooperative Block Load and Block Store constructs
for Python CUDA kernel DSLs. The first backend targets Numba-CUDA-MLIR and lowers
the operations to CUB.

The distribution is a universal Python wheel containing a coherent snapshot of
CUB, Thrust, libcu++, and CUDAX headers. Compilation always uses that bundled
snapshot; it does not substitute CUB headers from the active CUDA Toolkit.

## Installation

Choose the extra matching the CUDA Toolkit major version:

```bash
python -m pip install "cuda-coop[numba-cuda-mlir-cu12]"
python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
```

Python 3.10 through 3.14 is supported. Importing the portable `cuda.coop`
package does not require loading a compiler backend.

When using the portable namespace with Numba-CUDA-MLIR, import the compiler
runtime first so `cuda.coop` can activate its compiler hooks automatically:

```python
from numba_cuda_mlir import cuda

from cuda import coop
```

A standalone `cuda.coop` import does not discover or load optional compiler
runtimes or CUDA bindings. If `cuda.coop` was imported first, explicitly import
the qualified backend before compiling a kernel:

```python
from cuda import coop
import cuda.coop.numba_mlir  # Activate support for portable coop calls.
```

Using `import cuda.coop.numba_mlir as coop` instead activates the backend and
selects its qualified namespace.

## Block Load and Store

The portable `cuda.coop` entry points and the qualified
`cuda.coop.numba_mlir` entry points have matching signatures. The following is
a kernel-body example, where `source`, `destination`, and `count` are kernel
arguments:

```python
from numba_cuda_mlir import types

from cuda import coop

block = coop.this_block()
items = coop.ThreadData(2, dtype=types.int32)
loaded = coop.load(
    block,
    source,
    items,
    algorithm="direct",
    valid_items=count,
    oob_default=0,
)
coop.store(
    block,
    destination,
    loaded,
    algorithm="direct",
    valid_items=count,
)
```

`load` returns the same output object passed by the caller. `valid_items` counts
items across the complete block tile, while `offset` is a nonnegative element
offset. Runtime offsets are caller-validated. Source and destination arrays
must be one-dimensional and contiguous. Without `oob_default`, invalid Load
slots retain their previous values.

The public Block Load and Store enums retain the complete CUB algorithm
vocabulary. This release executes only `DIRECT`; selecting another member is
rejected by the Numba-CUDA-MLIR capability layer before provider compilation.
Load and Store currently support only `this_block()` even though the portable
API exposes the broader thread-group descriptor vocabulary.

## Temporary storage

Both operations may allocate temporary storage implicitly or accept a caller
descriptor:

```python
storage = coop.TempStorage(
    size_in_bytes=None,
    alignment=None,
    auto_sync=None,
    sharing="shared",
)
coop.load(block, source, items, temp_storage=storage)
```

Shared storage reuses compatible slices and synchronizes automatically by
default. Exclusive storage receives a distinct slice and cannot request
automatic synchronization.

These APIs are compile-time kernel constructs. Calling them outside a
compatible compiler context reports a structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/python/coop.html) for
the complete signatures.

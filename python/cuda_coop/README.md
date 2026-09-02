# `cuda.coop`

`cuda.coop` provides cooperative CUDA primitives for Python kernel DSLs. The
initial Numba-CUDA-MLIR integration reduces one scalar per CUDA thread-block
member through CUB BlockReduce.

Install the extra matching the CUDA Toolkit major version:

```console
pip install "cuda-coop[numba-cuda-mlir-cu12]"  # CUDA 12
pip install "cuda-coop[numba-cuda-mlir-cu13]"  # CUDA 13
```

The portable root API and the qualified Numba-CUDA-MLIR API have the same
BlockReduce contract. This kernel uses the root API:

```python
from numba_cuda_mlir import cuda

from cuda import coop


@cuda.jit
def block_sum(source, output):
    thread = cuda.threadIdx.x
    total = coop.sum(coop.this_block(), source[thread])
    if thread == 0:
        output[0] = total
```

Use `import cuda.coop.numba_mlir as coop` for an explicit qualified import.
Both forms lower to the same provider.

Every thread in the block must participate in converged control flow. The
result is defined only for block rank zero, so other threads must not consume
it. `valid_items` selects the prefix `[0, valid_items)` and must be between 1
and the block size and uniform across all block threads; all block threads
still invoke the collective.

The supported scalar dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit
integers plus `float32` and `float64`. `reduce` supports `sum`, `multiplies`,
`min`, `max`, `bit_and`, `bit_or`, and `bit_xor`; bitwise operations require an
integer dtype. Common aliases such as `+`, `add`, `minimum`, and `maximum` are
accepted. `sum` is the dedicated sum form, and an omitted `binary_op` on
`reduce` also selects sum.

The optional algorithms are `raking_commutative_only`, `raking`, and
`warp_reductions`; the default is `warp_reductions`. This initial slice accepts
scalar inputs and built-in operators only.

Importing `cuda.coop` succeeds without a compiler backend. A reduction outside
a compatible compiler context reports an explicit error.

See the [CCCL documentation](https://nvidia.github.io/cccl/python.html) for the
full contract and examples.

# `cuda.coop`

`cuda.coop` provides cooperative CUDA primitives for Python kernel DSLs. Its
portable API describes CUDA thread groups, per-thread data, temporary storage,
and collectives without importing a compiler at module load time.

```python
import numpy as np

from cuda import coop

block = coop.this_block()
items = coop.ThreadData(2, dtype=np.int32)
loaded = coop.load(block, source, items, valid_items=count, oob_default=0)
total = coop.sum(block, loaded)
coop.store(block, destination, loaded)
```

The common surface includes load/store, reduce/scan, exchange/shuffle,
adjacent-difference and discontinuity, histogram and run-length decode,
merge/radix sorting, radix rank, and TopK. CUTLASS Python DSL and
Numba-CUDA-MLIR integrations lower the same root calls in their compiler
contexts.

These functions are compile-time kernel constructs. Importing `cuda.coop`
succeeds without a compiler backend. A backend-dependent call outside a
compatible compiler context reports a structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/python.html) for the
supported signatures and examples.

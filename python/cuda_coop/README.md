# `cuda.coop`

`cuda.coop` provides cooperative CUDA primitives for Python kernel DSLs. Its
portable API describes CUDA thread groups, per-thread data, temporary storage,
and collectives through a backend-neutral root namespace.

Inside a kernel body, where `source`, `destination`, and `count` are kernel
arguments:

```python
import numpy as np

from cuda import coop

block = coop.this_block()
items = coop.ThreadData(2, dtype=np.int32)
loaded = coop.load(block, source, items, valid_items=count, oob_default=0)
total = coop.sum(block, loaded)
coop.store(block, destination, loaded, valid_items=count)
```

The initial common surface includes load/store and reduce/scan. Compatible
compiler integrations lower the same root calls in their compiler contexts.

These functions are compile-time kernel constructs. Importing `cuda.coop`
succeeds without a compiler backend. By default, the import also probes for an
installed compatible CUTLASS DSL and activates it when found. Set
`CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1` before importing `cuda.coop` to
skip automatic compiler probing and register integrations explicitly. A
backend-dependent call outside a compatible compiler context reports a
structured context error.

See the [CCCL documentation](https://nvidia.github.io/cccl/python.html) for the
supported signatures and examples.

# `cuda.coop`

`cuda.coop` provides cooperative CUDA primitives for Python kernel DSLs. The
initial API is deliberately small: one CUDA thread block can load and store a
fixed-size payload owned by each thread.

```python
import numpy as np

from cuda import coop

block = coop.this_block()
items = coop.ThreadData(2, dtype=np.int32)
loaded = coop.load(block, source, items)
coop.store(block, destination, loaded)
```

These functions are compile-time kernel constructs. The common root activates a
compatible installed CUTLASS DSL through concrete capability checks. Importing
`cuda.coop` still succeeds when CUTLASS is absent. Applications can also select
the backend explicitly with `import cuda.coop.cutlass as coop`.

Install the CUDA 13 CUTLASS backend dependencies with:

```console
python -m pip install "cuda-coop[cutlass]"
```

See the [CCCL documentation](https://nvidia.github.io/cccl/python.html) for the
supported signatures and examples.

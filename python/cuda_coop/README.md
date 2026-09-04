# `cuda.coop`

`cuda.coop` provides cooperative CUDA primitives for Python kernel DSLs. The
initial API is deliberately small: one CUDA thread block can load and store a
fixed-size payload owned by each thread. This first thin slice intentionally
supports the DIRECT load/store algorithm only.

```python
import numpy as np

from cuda import coop

block = coop.this_block()
items = coop.ThreadData(2, dtype=np.int32)
loaded = coop.load(block, source, items)
coop.store(block, destination, loaded)
```

These functions are compile-time kernel constructs. The common root selects a
compatible installed CUTLASS DSL only while that compiler's exact environment
manager is current. Importing `cuda.coop` still succeeds when CUTLASS is absent.
Applications can name the backend explicitly with
`import cuda.coop.cutlass as coop`; lowering still requires a compatible
CUTLASS compiler context.

Install the CUDA 13 CUTLASS backend dependencies with:

```console
python -m pip install "cuda-coop[cutlass]"
```

See the [CCCL documentation](https://nvidia.github.io/cccl/unstable/python/) for the
supported signatures and examples.

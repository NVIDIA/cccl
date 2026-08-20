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

These functions are compile-time kernel constructs. Importing `cuda.coop`
succeeds without a compiler backend; backend-dependent operations report a
structured compiler-context error until a compatible backend is active.

See the [CCCL documentation](https://nvidia.github.io/cccl/python.html) for the
supported signatures and examples.

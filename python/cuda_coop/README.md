# `cuda.coop`

`cuda.coop` provides cooperative CUDA primitives for Python kernel DSLs. The
initial API reduces one scalar per CUDA thread-block member through CUB
BlockReduce.

```python
from cuda import coop

block = coop.this_block()
total = coop.sum(block, value)
if thread_index == 0:
    output[0] = total
```

Every thread in the block participates. The result is defined only for block
rank zero. `reduce` supports built-in sum, multiplication, minimum, maximum,
and bitwise operators; both `reduce` and `sum` accept an optional valid prefix
and deterministic CUB algorithm selector.

These functions are compile-time kernel constructs. Importing `cuda.coop`
succeeds without a compiler backend; a reduction reports a structured
compiler-context error until a compatible backend is active.

See the [CCCL documentation](https://nvidia.github.io/cccl/python.html) for the
supported signatures and examples.

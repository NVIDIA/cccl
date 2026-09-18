# `cuda.coop`

`cuda.coop` provides cooperative Load and Store operations for Python GPU
kernels. The base `cuda-coop` package contains the shared API, Block and Warp
planning, and the CCCL headers used by compiler integrations.

```bash
python -m pip install cuda-coop
```

```python
from cuda import coop

block = coop.this_block()
warp = coop.this_warp()
logical_warp = warp.group_by(8)
```

Compiler backends implement `coop.load` and `coop.store` inside GPU kernels.
The shared API records the group, data type, algorithm, bounds, and temporary
storage requirements. It supports all Block Load/Store algorithms and physical
or logical Warp Load/Store. Importing the package and creating group
descriptors do not require a CUDA device.

Block operations accept `direct`, `striped`, `vectorize`, `transpose`,
`warp_transpose`, and `warp_transpose_timesliced`. Warp operations accept
`direct`, `striped`, `vectorize`, and `transpose`.

`load` fills its `ThreadData` output in place and returns `None`. `store`
writes per-thread values or a `ThreadData` payload. Both accept an element
`offset` and a `valid_items` count relative to the group's tile.

The package is experimental and its API is subject to change. See the
[CCCL Python documentation](https://nvidia.github.io/cccl/python/coop.html)
for the API and data-layout guidance.

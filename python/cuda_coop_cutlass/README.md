# `cuda.coop.cutlass`

Cooperative CUDA group primitives for kernels written with the CUTLASS CuTe DSL.
The `cuda-coop-cutlass` distribution supplies the `cuda.coop.cutlass` namespace
and its matching CCCL headers.

```python
import cuda.coop.cutlass as coop
```

The compiler integration targets CUTLASS DSL 4.9 with external LTO-IR linking,
exact launch facts and scoped trace finalization. The current draft is tested
with a retained 4.9 compiler candidate; released-wheel installation and
qualification are pending.

See the [documentation](https://cccl.nvidia.com/trentn/cuda-coop-cutlass/)
for the Programming Guide, API reference, and Developer Guide.

# `cuda.coop.cutlass`

Cooperative CUDA group primitives for kernels written with the CUTLASS CuTe DSL.
The `cuda-coop-cutlass` distribution supplies the `cuda.coop.cutlass` namespace
and its matching CCCL headers.

On Linux with a CUDA 13-capable NVIDIA driver, install a published release
with its CUDA Toolkit libraries:

```sh
python -m pip install 'cuda-coop-cutlass[cu13]'
```

```python
import cuda.coop.cutlass as coop
```

The compiler integration requires CUTLASS DSL 4.9.0.dev1 or later in the 4.x
series, with its CUDA 13 libraries. The package selects these dependencies
for external LTO-IR linking, exact launch facts and scoped trace finalization.

See the [documentation](https://nvidia.github.io/cccl/unstable/python/coop_cutlass/)
for the Programming Guide, API reference, and Developer Guide.

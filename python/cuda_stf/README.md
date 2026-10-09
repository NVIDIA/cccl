# CUDA STF Python Package

[`cuda.stf._experimental`](https://nvidia.github.io/cccl/python/stf.html)
provides Python bindings to **CUDASTF (Sequential Task Flow)**: you define logical
data and submit tasks that read or write that data, and STF infers the
dependencies and orchestrates execution and data movement. It is part of the
[CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cpp.html#cccl-cpp-libraries).

The API is exposed under the `_experimental` subpackage because it is still
evolving and may change without notice. CUDASTF is currently **Linux-only**.

## Installation

Install from PyPI:

```bash
pip install cuda-stf[cu13]  # For CUDA 13.x (pip-installed cuda-toolkit)
pip install cuda-stf[cu12]  # For CUDA 12.x (pip-installed cuda-toolkit)
```

If you already have a CUDA toolkit on your system and do not want pip to
install it, use the `sysctk` variants:

```bash
pip install cuda-stf[sysctk13]  # For CUDA 13.x (system CUDA toolkit)
pip install cuda-stf[sysctk12]  # For CUDA 12.x (system CUDA toolkit)
```

For a smaller install without Numba (when you drive kernels through
`cuda.core` / `cuda.compute` or your own launches), use the `minimal-*`
variants:

```bash
pip install cuda-stf[minimal-cu13]       # pip CUDA toolkit, no Numba
pip install cuda-stf[minimal-sysctk13]   # system CUDA toolkit, no Numba
```

Install `cuda-cccl` as well when using `cuda.compute` with STF or compiling
external C++ code against the cudax headers; it supplies the libcudacxx, CUB,
and Thrust headers.

Feature dependencies are installed separately as needed: `cuda-cccl`
(`cuda.compute` and header discovery), `numba` / `numba-cuda` (Numba interop,
bundled by the non-`minimal` extras), `cupy`, `torch` (PyTorch interop),
`warp-lang` (Warp interop), and `nvmath-python` (cuBLAS/cuSOLVER examples).

### Install from source (Linux only)

```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_stf
pip install -e .[test-cu13]  # or .[test-cu12], .[test-sysctk13], .[test-sysctk12]
```

Building from source compiles the native `cccl.c.experimental.stf` / `cudax`
extension, so a C++ toolchain and CMake (`>=3.30`) with Ninja are required in
addition to the CUDA toolkit. The `test-*` extras add `cuda-cccl`, `pytest`,
`pytest-xdist`, and CuPy so the test suite (`pytest tests/`) can run.

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with
Compute Capability 7.5+, Linux.

## Device memory interchange: CAI and DLPack

`DeviceArray` (buffers allocated through a `data_place`, including composite
localized places) implements **both** interchange protocols. They are
complementary — a consumer picks the semantics by construction:

- **CUDA Array Interface** (`__cuda_array_interface__`, CAI v3) *describes*
  the memory and transfers **no ownership**: the `DeviceArray` must outlive
  every borrowed view. This is the zero-copy path used by task arguments,
  Numba, `cuda.compute`, and `torch.as_tensor`.
- **DLPack** (`__dlpack__` / `__dlpack_device__`) *carries ownership*: the
  exported capsule keeps the array alive and the consumer's deleter releases
  it, so e.g. `torch.from_dlpack(arr)` yields a tensor whose storage lifetime
  owns the allocation. Deallocation stays with the `DeviceArray` finalizer —
  one deallocation point regardless of protocol.

```python
stf.machine_init()
arr = stf.DeviceArray((4, 8), np.float32, stf.data_place.device(0))
borrowed = numba.cuda.as_cuda_array(arr)   # CAI: arr must stay alive
owned    = torch.from_dlpack(arr)          # DLPack: the tensor keeps it alive
```

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python/stf.html](https://nvidia.github.io/cccl/python/stf.html)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf](https://github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf)

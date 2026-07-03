# CUDA STF Python Package

[`cuda.stf._experimental`](https://nvidia.github.io/cccl/python/stf.html)
provides Python bindings to **CUDASTF (Stream Task Flow)**: you define logical
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

`cuda-stf` is self-contained: it ships its own STF/cudax headers and CUDA
version detection, so it has no hard dependency on `cuda-cccl`. Installing
`cuda-cccl` alongside it is optional and only needed to compile external C++
code against the cudax headers (it supplies the lower-level
libcudacxx/CUB/Thrust headers).

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with
Compute Capability 7.5+, Linux.

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python/stf.html](https://nvidia.github.io/cccl/python/stf.html)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf](https://github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf)

# CUDA CCCL Python Package

[`cuda.cccl`](https://nvidia.github.io/cccl/python)
provides a Pythonic interface to the
[CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cpp.html#cccl-cpp-libraries).
It provides the following modules:

- **`cuda.compute`** - Device-level parallel algorithms (reduce, scan, sort, etc.) and iterators
- **`cuda.coop`** - Block and warp-level cooperative primitives for custom CUDA kernels

## Installation

Install from PyPI:

```bash
pip install cuda-cccl[cu13]  # For CUDA 13.x (pip-installed cuda-toolkit)
pip install cuda-cccl[cu12]  # For CUDA 12.x (pip-installed cuda-toolkit)
```

If you already have a CUDA toolkit on your system and do not want pip to
install it, use the `sysctk` variants:

```bash
pip install cuda-cccl[sysctk13]  # For CUDA 13.x (system CUDA toolkit)
pip install cuda-cccl[sysctk12]  # For CUDA 12.x (system CUDA toolkit)
```

For a minimal install without Numba (useful when supplying pre-compiled operators):

```bash
pip install cuda-cccl[minimal-cu13]      # pip-installed cuda-toolkit
pip install cuda-cccl[minimal-sysctk13]  # system CUDA toolkit
```

Install from conda-forge:

```bash
conda install -c conda-forge cccl-python
```

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with Compute Capability 6.0+

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python](https://nvidia.github.io/cccl/python)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/compute/examples](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/compute/examples) and [github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/coop/examples](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/coop/examples)

# cuda-compute

The `cuda.compute` library provides composable primitives for building custom
parallel algorithms on the GPU—without writing CUDA kernels directly. It
includes device-level parallel algorithms (reduce, scan, sort, etc.) and
iterators.

## Installation

Install from PyPI:

```bash
pip install "cuda-compute[cu13]"  # For CUDA 13.x (pip-installed cuda-toolkit)
pip install "cuda-compute[cu12]"  # For CUDA 12.x (pip-installed cuda-toolkit)
```

If you already have a CUDA toolkit on your system and do not want pip to
install it, use the `sysctk` variants:

```bash
pip install "cuda-compute[sysctk13]"  # For CUDA 13.x (system CUDA toolkit)
pip install "cuda-compute[sysctk12]"  # For CUDA 12.x (system CUDA toolkit)
```

For a minimal install without Numba (useful when supplying pre-compiled operators):

```bash
pip install "cuda-compute[minimal-cu13]"      # pip-installed cuda-toolkit
pip install "cuda-compute[minimal-sysctk13]"  # system CUDA toolkit
```

When developing from a source checkout, install this project and its sibling
header project together so the exact dependency is resolved locally:

```bash
pip install -e ../cccl_headers -e ".[test-cu13]"
```

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with Compute Capability 7.5+

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python](https://nvidia.github.io/cccl/python)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_compute/tests/compute/examples](https://github.com/NVIDIA/cccl/tree/main/python/cuda_compute/tests/compute/examples)

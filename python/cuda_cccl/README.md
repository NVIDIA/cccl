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
pip install cuda-cccl[cu13]  # For CUDA 13.x
pip install cuda-cccl[cu12]  # For CUDA 12.x
```

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with Compute Capability 6.0+

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python](https://nvidia.github.io/cccl/python)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/compute/examples](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/compute/examples) and [github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/coop/examples](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/coop/examples)

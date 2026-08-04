# cuda-compute

`cuda-compute` provides the `cuda.compute` Python interface to the
[CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cpp.html#cccl-cpp-libraries).
It includes device-level parallel algorithms (reduce, scan, sort, etc.) and
iterators. Its required CCCL headers are private implementation data: wheel
builds package them directly from the canonical CCCL repository trees.

The distribution and import names are intentionally different:

| Distribution | Python import | Purpose |
| --- | --- | --- |
| `cuda-compute` | `cuda.compute` | Device-level algorithms and iterators |
| `cuda-cccl` | None | Aggregate metapackage |

## Installation

Install from PyPI:

```bash
pip install cuda-compute[cu13]  # For CUDA 13.x (pip-installed cuda-toolkit)
pip install cuda-compute[cu12]  # For CUDA 12.x (pip-installed cuda-toolkit)
```

If you already have a CUDA toolkit on your system and do not want pip to
install it, use the `sysctk` variants:

```bash
pip install cuda-compute[sysctk13]  # For CUDA 13.x (system CUDA toolkit)
pip install cuda-compute[sysctk12]  # For CUDA 12.x (system CUDA toolkit)
```

For a minimal install without Numba (useful when supplying pre-compiled operators):

```bash
pip install cuda-compute[minimal-cu13]      # pip-installed cuda-toolkit
pip install cuda-compute[minimal-sysctk13]  # system CUDA toolkit
```

The `cuda-cccl` metapackage forwards the same extras, so existing aggregate
installs such as `pip install cuda-cccl[cu13]` continue to install this package.

When developing from a complete CCCL checkout, install from the repository
root. Editable mode uses the canonical `libcudacxx`, CUB, and Thrust
directories in place. It does not copy headers into `python/cuda_compute`:

```bash
pip install -e "./python/cuda_compute[test-cu13]"
```

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with Compute Capability 7.5+

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python](https://nvidia.github.io/cccl/python)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_compute/tests/compute/examples](https://github.com/NVIDIA/cccl/tree/main/python/cuda_compute/tests/compute/examples)

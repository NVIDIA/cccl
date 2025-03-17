## CUDA Experimental: Library for experimental features in CUDA Core Compute Libraries.
CUDA Experimental serves as a distribution channel for features that are considered experimental in the CUDA Core Compute Libraries.
Some of them are still actively designed or developed and their API is evolving.
Some of them are specific to one hardware architecture and are still looking for a generic and forward compatible exposure.
Finally, some of them need to prove useful enough to deserve long term support.

**All APIs available in CUDA Experimental are not considered stable and can change without a notice.** They can also be deprecated or removed on a much faster cadence than in other CCCL libraries.

Features are exposed here for the CUDA C++ community to experiment with and provide feedback on how to shape it to best fit their use cases.
Once we become confident a feature is ready and would be a great permanent addition in CCCL, it will become a part of some other CCCL library with a stable API.

## Installation
CUDA Experimental library is **not** distributed with the CUDA Toolkit like the rest of CCCL. It is only available on the [CCCL GitHub repository](https://github.com/NVIDIA/cccl).

CUDA Experimental compilation requires C++17 standard or newer. Supported compilers are:

CUDA Compilers:
- NVCC 12.3+

NVCC host compilers:
- GCC 7+
- Clang 9+
- MSVC 2019+

Everything in CUDA Experimental is header-only, so cloning and including it in a simple project is as easy as the following:
```bash
git clone https://github.com/NVIDIA/cccl.git
# Note:
nvcc -Icccl/cudax/include main.cu -o main
```

A CMake target `cudax::cudax` is available as part of the CCCL package when `CCCL_ENABLE_UNSTABLE` is set to a truthy value before calling `find_package` or `add_subdirectory`.

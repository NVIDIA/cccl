.. _infra-cmake-architecture-flags:

Architecture flags
==================

CCCL builds device code for the SM architectures named in ``CMAKE_CUDA_ARCHITECTURES``, a
semicolon-separated list of SM numbers, each optionally tagged ``-real`` (embed SASS) or
``-virtual`` (embed PTX for JIT on newer GPUs). CMake also accepts ``native``, ``all``, and
``all-major``. The
`CUDA_ARCHITECTURES property <https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html>`_
documents the standard syntax and values.

CCCL adds two values, expanded at configure time by ``cccl_check_cuda_architectures()`` against
the toolkit found at configure time:

- ``all-cccl`` — every architecture the current nvcc supports at or above CCCL's minimum.
- ``all-major-cccl`` — one entry per major architecture at or above the minimum, carrying
  forward PTX on the highest.

Both resolve against the installed toolkit rather than a fixed table. The minimum supported
architecture and the full expansion logic live in ``cmake/CCCLCheckCudaArchitectures.cmake``.
Expansion runs only for a top-level CCCL build; downstream consumers pass a concrete list or
``native``.

.. _cccl-cpp-libraries:

CUDA C++ Core Libraries
=======================

.. toctree::
   :hidden:
   :maxdepth: 3

   libcudacxx/index
   cub/index
   thrust/index
   cudax/index
   cccl/tma
   cccl/3.0_migration_guide
   cccl/development/index
   cccl/contributing
   cccl/license

Welcome to the CUDA Core Compute Libraries (CCCL) libraries for C++.

The concept for the  CCCL C++ librarires grew organically out of the Thrust,
CUB, and libcudacxx projects that were developed independently over the years
with a similar goal: to provide high-quality, high-performance, and
easy-to-use C++ abstractions for CUDA developers. Naturally, there was a lot
of overlap among the three projects, and it became clear the community would
be better served by unifying them into a single repository.

- :doc:`libcu++ <libcudacxx/index>`
  is the CUDA C++ Standard Library. It provides an implementation of the C++
  Standard Library that works in both host and device code. Additionally, it
  provides abstractions for CUDA-specific hardware features like
  synchronization primitives, cache control, atomics, and more.

- :doc:`CUB <cub/index>`
  is a lower-level, CUDA-specific library designed for speed-of-light parallel
  algorithms across all GPU architectures. In addition to device-wide
  algorithms, it provides *cooperative algorithms* like block-wide reduction
  and warp-wide scan, providing CUDA kernel developers with building blocks to
  create speed-of-light, custom kernels.

- :doc:`Thrust <thrust/index>`
  is the C++ parallel algorithms library which inspired the introduction of
  parallel algorithms to the C++ Standard Library. Thrust's high-level
  interface greatly enhances programmer productivity while enabling performance
  portability between GPUs and multicore CPUs via configurable backends that
  allow using multiple parallel programming frameworks (such as CUDA, TBB, and
  OpenMP).

- :doc:`Cuda Experimental <cudax/index>`
  is a library of experimental features that are still in the design process.

The main goal of the CCCL C++ libraries is to fill a similar role that the
Standard C++ Library fills for Standard C++: provide general-purpose,
speed-of-light tools to CUDA C++ developers, allowing them to focus on
solving the problems that matter. Unifying these projects is the first step
towards realizing that goal.

CUDA C++ Core Libraries
=======================

.. toctree::
   :hidden:
   :maxdepth: 3

   libcu++ <https://nvidia.github.io/cccl/libcudacxx/>
   CUB <https://nvidia.github.io/cccl/cub/>
   Thrust <https://nvidia.github.io/cccl/thrust/>
   Cuda Experimental <https://nvidia.github.io/cccl/cudac/>

Welcome to the CUDA C++ Core Libraries (CCCL) where our mission is to make CUDA C++ more delightful.

The concept for the CUDA C++ Core Libraries (CCCL) grew organically out of the Thrust, CUB, and libcudacxx projects that were developed independently over the years with a similar goal: to provide high-quality, high-performance, and easy-to-use C++ abstractions for CUDA developers.
Naturally, there was a lot of overlap among the three projects, and it became clear the community would be better served by unifying them into a single repository.

- `libcu++ <https://nvidia.github.io/cccl/libcudacxx/>`__ is the CUDA C++ Standard Library. It provides an implementation of the C++ Standard Library that works in both host and device code. Additionally, it provides abstractions for CUDA-specific hardware features like synchronization primitives, cache control, atomics, and more.

- `CUB <https://nvidia.github.io/cccl/cub/>`__ is a lower-level, CUDA-specific library designed for speed-of-light parallel algorithms across all GPU architectures. In addition to device-wide algorithms, it provides *cooperative algorithms* like block-wide reduction and warp-wide scan, providing CUDA kernel developers with building blocks to create speed-of-light, custom kernels.

- `Thrust <https://nvidia.github.io/cccl/thrust/>`__ is the C++ parallel algorithms library which inspired the introduction of parallel algorithms to the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs via configurable backends that allow using multiple parallel programming frameworks (such as CUDA, TBB, and OpenMP).

- `Cuda Experimental <https://nvidia.github.io/cccl/cudax/>`__ is a library of exerimental features that are still in the design process.

The main goal of CCCL is to fill a similar role that the Standard C++ Library fills for Standard C++: provide general-purpose, speed-of-light tools to CUDA C++ developers, allowing them to focus on solving the problems that matter.
Unifying these projects is the first step towards realizing that goal.

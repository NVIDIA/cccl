.. _thrust-module:

Thrust
======

.. toctree::
   :hidden:
   :maxdepth: 2

   Overview <self>
   developer_overview
   releases
   release_process
   API documentation <api>
   API reference <api/index>

Thrust is the C++ parallel algorithms library which inspired the introduction of parallel algorithms to the
C++ Standard Library. Thrust's **high-level** interface greatly enhances programmer **productivity** while
enabling performance portability between GPUs and multicore CPUs.
It builds on top of established parallel programming frameworks (such as CUDA, TBB, and OpenMP).
It also provides a number of general-purpose facilities similar to those found in the C++ Standard Library.

Thrust is an open source project; it is available on `GitHub <https://github.com/NVIDIA/cccl>`__
and included in the NVIDIA HPC SDK and CUDA Toolkit.
If you have one of those SDKs installed, no additional installation or compiler flags are needed to use Thrust.

Examples
--------

Thrust is best learned through examples.

-------------
CMake Example
-------------

A complete, standalone example project showing how to write a CMake build system that uses Thrust with any supported
device system is available in the CCCL repository `here <https://github.com/NVIDIA/cccl/tree/main/examples/thrust_flexible_device_system>`__.

------------------
Thrust API Example
------------------

The following example generates random numbers serially and then transfers them
  to a parallel device where they are sorted.

.. code:: cpp

   #include <thrust/host_vector.h>
   #include <thrust/device_vector.h>
   #include <thrust/generate.h>
   #include <thrust/sort.h>
   #include <thrust/copy.h>
   #include <thrust/random.h>

   int main() {
   // Generate 32M random numbers serially.
   thrust::default_random_engine rng(1337);
   thrust::uniform_int_distribution<int> dist;
   thrust::host_vector<int> h_vec(32 << 20);
   thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

   // Transfer data to the device.
   thrust::device_vector<int> d_vec = h_vec;

   // Sort data on the device.
   thrust::sort(d_vec.begin(), d_vec.end());

   // Transfer data back to host.
   thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
   }


`See it on Godbolt <https://godbolt.org/z/GeWEd8Er9>`__

This example demonstrates computing the sum of some random numbers in parallel:

.. code:: cpp

   #include <thrust/host_vector.h>
   #include <thrust/device_vector.h>
   #include <thrust/generate.h>
   #include <thrust/reduce.h>
   #include <thrust/functional.h>
   #include <thrust/random.h>

   int main() {
   // Generate random data serially.
   thrust::default_random_engine rng(1337);
   thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
   thrust::host_vector<double> h_vec(32 << 20);
   thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

   // Transfer to device and compute the sum.
   thrust::device_vector<double> d_vec = h_vec;
   double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
   }

`See it on Godbolt <https://godbolt.org/z/cnsbWWME7>`__

Getting The Thrust Source Code & Developing Thrust
-----------------------------------------------------

Thrust started as a stand-alone project, but as of March 2024 Thrust is a part of the
CUDA Core Compute Libraries (CCCL). Please refer to the
`CCCL Getting Started section <https://github.com/NVIDIA/cccl?tab=readme-ov-file#getting-started>`__ and the
`Contributing Guide: <https://github.com/NVIDIA/cccl/blob/main/CONTRIBUTING.md>`__ for instructions on how to
get started developing the CCCL sources.

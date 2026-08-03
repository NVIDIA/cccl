CCCL Python Libraries
======================

Overview
--------

The CUDA Core Compute Libraries (CCCL) for Python are a collection of modules
with the shared goal of providing **high-quality, high-performance, and easy-to-use**
abstractions for CUDA Python developers.

* :doc:`cuda.compute <compute/index>` — Composable device-level primitives for building
  custom parallel algorithms, without writing CUDA kernels directly.
* :doc:`cuda.cccl.headers <headers_api>` — Programmatic access to the bundled
  libcu++, CUB, Thrust, and CUDAX headers and their relocatable CMake packages.

These import packages are distributed separately. ``cuda-compute`` provides
``cuda.compute``, ``cccl-headers`` provides ``cuda.cccl.headers``, and the
``cuda-cccl`` metapackage installs the complete supported set.

These libraries expose the generic, highly-optimized algorithms from the
`CCCL C++ libraries <https://nvidia.github.io/cccl/cpp.html>`_,
which have been tuned to provide optimal performance across GPU architectures.

Who is this for?
----------------

- **Library authors** building parallel algorithms that need portable performance
  across GPU architectures—without dropping to CUDA C++.

- **Application developers** using PyTorch, CuPy, or other GPU-accelerated frameworks
  who need custom algorithms beyond what those libraries provide.

.. toctree::
   :maxdepth: 2
   :caption: CCCL Python Libraries

   setup
   compute/index
   resources
   api_reference

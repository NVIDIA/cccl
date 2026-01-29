CCCL Python Libraries
======================

Overview
--------

The CUDA Core Compute Libraries (CCCL) for Python are a collection of modules
with the shared goal of providing **high-quality, high-performance, and easy-to-use**
abstractions for CUDA Python developers.

* :doc:`cuda.compute <compute>` — Composable device-level primitives for building
  custom parallel algorithms, without writing CUDA kernels directly.

* :doc:`cuda.coop <coop>` — Cooperative block- and warp-level algorithms for
  writing highly efficient CUDA kernels with `Numba CUDA <https://nvidia.github.io/numba-cuda/>`_.

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
   compute
   coop
   resources
   api_reference

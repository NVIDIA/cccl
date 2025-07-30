CCCL Python Libraries
======================

Welcome to the CCCL Python libraries! This collection provides Pythonic interfaces to
CUDA Core Compute Libraries (CCCL).

The CCCL Python libraries consist of two main components:

:doc:`cuda.cccl.parallel <parallel>`
  Device-level parallel algorithms for operations on entire arrays or data ranges.

:doc:`cuda.cccl.cooperative <cooperative>`
  Block and warp-level cooperative algorithms for building custom CUDA kernels with Numba.

.. toctree::
   :maxdepth: 1

   setup
   parallel
   cooperative
   resources
   api_reference

.. _cccl-python-coop-numba-mlir:

Getting Started with ``cuda.coop.numba_mlir``
=============================================

The Numba-CUDA-MLIR backend lets ``numba_cuda_mlir`` kernels use the portable
:mod:`cuda.coop` operations. When ``cuda.coop`` is imported with a compatible
``numba-cuda-mlir`` installation, the backend registers whole-function
rewrites with the compiler; cooperative calls inside an ``@cuda.jit`` kernel
are then recognized during compilation and replaced with compiled CUB
collectives linked as LTO-IR. Importing :mod:`cuda.coop.numba_mlir` activates
the same integration explicitly.

The first primitives on the portable contract are block and warp load/store
with partial-tile controls; the following commits grow the same rewrite stack
to the full portable operation set. The backend additionally supports
``gpu_dataclass`` temp-storage traits, explicit shared-memory ``TempStorage``
planning with automatic synchronization, and scoped ``_block``/``_warp``
two-phase factories.

Installation
------------

Install the ``cuda-coop`` distribution with the Numba-CUDA-MLIR dependencies:

.. code-block:: bash

   pip install 'cuda-coop[numba-cuda-mlir]'

The backend verifies the compiler's whole-function planner hooks at import
time. A ``numba-cuda-mlir`` release without those hooks is reported through a
``CudaCoopAutoRegistrationWarning`` naming the missing capability, and the
portable root API stays importable.

Block load and store
--------------------

A complete CUDA thread block loads a tile from global memory into per-thread
registers and stores it back:

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda

   import cuda.coop.numba_mlir as coop

   @cuda.jit
   def copy_kernel(values_in, values_out):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       loaded = coop.load(block, values_in, items)
       coop.store(block, values_out, loaded)

See :doc:`coop_api` for the API reference.

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

All portable operation families are served at block scope, plus the
warp-scoped load/store, reduce, scan, exchange, and merge sort forms. The
backend additionally supports ``gpu_dataclass`` temp-storage traits, explicit
shared-memory ``TempStorage`` planning with automatic synchronization, and
scoped ``_block``/``_warp`` two-phase factories.

Installation
------------

Install the ``cuda-coop`` distribution with the Numba-CUDA-MLIR dependencies:

.. code-block:: bash

   pip install 'cuda-coop[numba-cuda-mlir]'

The backend verifies the compiler's whole-function planner hooks at import
time. A ``numba-cuda-mlir`` release without those hooks is reported through a
``CudaCoopAutoRegistrationWarning`` naming the missing capability, and the
portable root API stays importable.

Block sum
---------

This executable example reduces one value per thread across a CUDA thread
block:

.. literalinclude:: ../../python/cuda_coop/examples/numba_mlir/block_sum.py
   :language: python
   :caption: Block-wide sum through the Numba-CUDA-MLIR backend.

Run it from the repository root or an installed wheel:

.. code-block:: bash

   python -m examples.numba_mlir.block_sum

Further executable examples under ``python/cuda_coop/examples/numba_mlir``
cover load/scan/store pipelines, shared temp storage, radix sort pairs,
partial-tile top-k, group hierarchies, and the portable root API.

See :doc:`coop_api` for the complete API.

.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` provides compiler-neutral contracts for cooperative primitives
inside Python GPU kernels. The initial Numba-CUDA-MLIR integration describes a
CUDA thread block and reduces one scalar value per thread with CUB BlockReduce.

Installation
------------

Install the extra matching the CUDA Toolkit major version:

.. code-block:: console

   pip install "cuda-coop[numba-cuda-mlir-cu12]"  # CUDA 12
   pip install "cuda-coop[numba-cuda-mlir-cu13]"  # CUDA 13

The extras install ``numba-cuda-mlir>=0.5.0,<0.6`` and
``cuda-core>=0.5.1,<2`` together with the matching CUDA dependencies.

Numba-CUDA-MLIR example
-----------------------

The portable root API and :mod:`cuda.coop.numba_mlir` expose the same block
reduction contract. This kernel uses the root API:

.. code-block:: python

   from numba_cuda_mlir import cuda

   from cuda import coop


   @cuda.jit
   def block_sum(source, output):
       thread = cuda.threadIdx.x
       total = coop.sum(coop.this_block(), source[thread])
       if thread == 0:
           output[0] = total

Use ``import cuda.coop.numba_mlir as coop`` for an explicit qualified import.
Both spellings lower to the same provider. The compiler integration obtains the
exact block dimensions from the launch configuration, so the operation call
does not repeat them.

Participation and result contract
---------------------------------

Every thread in the block must invoke the collective in converged control
flow. The returned scalar is defined only for block rank zero. Other threads
must not consume it, and kernels must guard any use of the result accordingly.
There is no broadcast mode in this initial API.

The optional ``valid_items`` argument selects the block-rank prefix
``[0, valid_items)``. Its value must be between one and the block size,
inclusive, and uniform across all block threads. Every block thread still
invokes the collective, including threads outside the valid prefix.

Supported values and operations
-------------------------------

The integration supports one scalar value per thread with these dtypes:

* ``int8``, ``int16``, ``int32``, and ``int64``;
* ``uint8``, ``uint16``, ``uint32``, and ``uint64``; and
* ``float32`` and ``float64``.

:func:`cuda.coop.reduce` accepts these built-in operators:

* ``sum`` and its ``+``, ``add``, and ``plus`` aliases;
* ``multiplies`` and its ``*``, ``mul``, and ``multiply`` aliases;
* ``min`` and ``max``, including ``minimum`` and ``maximum``; and
* ``bit_and``, ``bit_or``, and ``bit_xor``, including ``&``, ``|``, and ``^``.

Bitwise reductions require an integer dtype. An omitted ``binary_op`` selects
sum, and :func:`cuda.coop.sum` is the dedicated sum form. Python callback
operators and per-thread array payloads are outside this initial slice.

Algorithm selection
-------------------

The optional ``algorithm`` argument selects one deterministic CUB BlockReduce
algorithm:

* ``raking_commutative_only``;
* ``raking``; or
* ``warp_reductions``.

The default is ``warp_reductions``. The algorithm and built-in operator are
compile-time selectors; ``valid_items`` may be supplied by the kernel at
runtime.

Backend activation
------------------

Importing :mod:`cuda.coop` does not make a compiler backend mandatory. When
Numba-CUDA-MLIR is installed, the root import activates its integration.
Calling a reduction outside a compatible compiler context raises a structured
compiler-context error when no backend is active. A qualified backend marker
raises a runtime error when called outside kernel compilation. Set
``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1`` before importing the root package
to disable automatic backend probing and use an explicit qualified import
instead.

API reference
-------------

See :doc:`coop_api` for the portable root API.

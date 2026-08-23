.. _cccl-python-coop-cutlass:

Getting Started with ``cuda.coop.cutlass``
===========================================

The CUTLASS backend lets CUTLASS CuTe DSL kernels use the portable
:mod:`cuda.coop` operations. A compatible CUTLASS compiler context activates
the backend automatically; the qualified :mod:`cuda.coop.cutlass` API selects
the same implementation explicitly. The backend renders each requested
collective as a CUB or CUDAX C++ shim, compiles it to LTO-IR with NVRTC
against the wheel's private CCCL header bundle, and attaches the artifact
during CUTLASS finalization. The backend supports CUDA 13.

The first primitives on the portable contract are block and warp load/store
with partial-tile ``valid_items``/``oob_default`` controls and element
offsets; the following commits grow the same provider stack to the full
portable operation set.

Installation
------------

Install the ``cuda-coop`` distribution with the CUTLASS dependencies:

.. code-block:: bash

   pip install 'cuda-coop[cutlass]'

An operation must be traced from a compatible compiler context. If no provider
can serve that context, ``cuda.coop`` reports the structured error below
instead of silently choosing another backend:

.. code-block:: text

   cuda.coop.<feature> requires an active compiler backend; install or import a compatible backend before compiling a kernel

CUDA thread block load and store
--------------------------------

A complete CUDA thread block loads a tile from contiguous, pointer-backed
memory into per-thread registers and stores it back. ``valid_items`` defines
the valid prefix of a partial tile; ``oob_default`` supplies a sentinel for
each tail item; ``offset`` is measured in elements. Every thread participates
even when its items fall in the tail.

.. code-block:: python

   import numpy as np

   from cuda import coop

   @cute.kernel
   def copy_kernel(source: cute.Tensor, destination: cute.Tensor):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       loaded = coop.load(block, source, items, valid_items=73, oob_default=-1)
       coop.store(block, destination, loaded, valid_items=73)

Use :mod:`cuda.coop.cutlass` directly when a kernel should be explicitly tied
to CUTLASS. Its exports have the same signatures and behavior as the
automatically activated portable root API.

See :doc:`coop_api` for the API reference.

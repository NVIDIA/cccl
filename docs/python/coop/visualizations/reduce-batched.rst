.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-reduce-batched:

Batched Warp Reduction
======================

These primitives are currently implemented by Numba-CUDA-MLIR. CUTLASS
does not yet implement them; see :ref:`backend coverage <coop-backends>`.

:func:`cuda.coop.reduce_batched` reduces several independent batches across
a warp. Each thread contributes one item to each batch: local slot zero
belongs to batch zero, slot one to batch one, and so on. The result has
one aggregate per batch, distributed among the threads.

This is useful when every lane holds several features for one sample and
the kernel needs a sum, minimum, or maximum for each feature. Ordinary
:func:`~cuda.coop.reduce` combines a group's payload items into one
aggregate; batched reduction retains the separate batch axis.

.. coop-visualization:: reduce-batched

   .. only:: html

      .. figure:: reduce-batched.svg
         :alt: Four lanes each hold three features. Their per-feature sums are 16, 20, and 24, returned in lanes zero, one, and two. Lane three has no result.
         :width: 100%

         Three batches across a logical warp of four lanes.

   Each column below is one independent reduction. ``?`` is an unspecified
   output slot, which the kernel must not read or store.

   .. code-block:: text

      Lane        Input slots (batches 0, 1, 2)    Returned slot
      L0                     1   2   3                  16
      L1                     3   4   5                  20
      L2                     5   6   7                  24
      L3                     7   8   9                   ?
      Batch sums            16  20  24

Input slots and output ownership
--------------------------------

For ``B`` batches and a warp of ``W`` threads, each input payload contains
``B`` items. Each returned payload contains ``ceil(B / W)`` slots.
``output_layout="striped"`` assigns batch ``lane + slot * W`` to each
result slot. ``output_layout="blocked"`` assigns batch
``lane * ceil(B / W) + slot`` instead. The input interpretation is the
same for both layouts.

Try ten batches with four lanes to see the distinction: striped lane
zero receives batches 0, 4, and 8; blocked lane zero receives batches 0,
1, and 2. The remaining two allocated output slots do not correspond to
any batch. Guard stores by batch index, rather than assuming every slot
in the returned payload is valid.

The explorer groups inputs by batch to explain the reduction axis. It
does not imply that CUB performs a shared-memory transpose. CUB provides
a native ``WarpReduceBatched`` collective; the diagrams describe its
result contract, not its compiled instruction order.

Using batched reduction in a kernel
-----------------------------------

This tested example sums three features across each physical warp. The
input is laid out as 32 consecutive samples per warp, with three
consecutive features per sample. Each warp writes three results.

.. code-block:: python

   from numba_cuda_mlir import cuda
   from cuda import coop

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_reduce_batched.py
   :language: python
   :start-after: # example-begin reduce-batched-features
   :end-before: # example-end reduce-batched-features
   :dedent: 4

The host test launches two warps and checks their feature sums separately.
For more than one block, include the block's input and output base offsets.
The primitive never combines results from different warps.

The Numba-CUDA-MLIR backend supports complete physical warps and logical
warps of 1, 2, 4, 8, or 16 threads. Every member of the selected warp must
participate, even if that member owns no result. Other logical warps may
take another branch. Temporary storage is allocated per warp by the
compiler; this operation does not accept a caller ``TempStorage``.

The batch count is a positive compile-time payload extent. The result
uses the input dtype without accumulator promotion. Built-in operator
strings match :func:`~cuda.coop.reduce`; the qualified
:func:`cuda.coop.numba_mlir.reduce_batched` form additionally accepts a
stateless device callback and fixed local-array inputs. Reduction
operators must be associative and commutative.

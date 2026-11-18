.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-reduce-batched:

Batched Warp Reduction
======================

:func:`cuda.coop.cutlass.reduce_batched` reduces several batches in one
call, across the lanes of a warp or of a group of lanes. Item 0 of every
lane belongs to batch 0, item 1 to batch 1, and so on. Each batch has its
own result, and the results are spread over the lanes.

A typical use: every lane holds several features of one sample, and the
kernel needs the sum, minimum or maximum of each feature.

.. coop-visualization:: reduce-batched

   .. only:: html

      .. figure:: reduce-batched.svg
         :alt: Four lanes each hold three features. Their per-feature sums are
            16, 20, and 24, returned in lanes zero, one, and two. Lane three
            has no result.
         :width: 100%

         Three batches across a group of four lanes.

   Each column below is one batch. ``?`` marks a returned item that has no
   batch. The kernel must not read or store it.

   .. code-block:: text

      Lane        Input items (batches 0, 1, 2)    Returned item
      L0                     1   2   3                  16
      L1                     3   4   5                  20
      L2                     5   6   7                  24
      L3                     7   8   9                   ?
      Batch sums            16  20  24

Batches and output layout
-------------------------

For ``B`` batches and a group of ``W`` lanes, each lane passes ``B`` items
and gets ``ceil(B / W)`` items back. With ``output_layout="striped"``, item
``i`` of the lane with rank ``r`` holds batch ``r + i * W``. With
``output_layout="blocked"``, it holds batch ``r * ceil(B / W) + i``. Both
layouts read the input the same way.

Try ten batches with four lanes. With the striped layout, lane 0 receives
batches 0, 4 and 8. With the blocked layout, it receives batches 0, 1
and 2. Two of the twelve returned items have no batch. The explorer calls
an item a slot and marks these two with ``?``. A kernel tests the batch
number before it reads or stores an item.

The explorer's second row groups the inputs by batch to show which values
each reduction combines. In a kernel, ``cub::WarpReduceBatched`` combines
the lanes' values in registers and uses no shared memory.

Using batched reduction in a kernel
-----------------------------------

This tested example sums ``items_per_thread`` features across each of the
two warps of a block. The input holds 32 samples per warp, one after
another, with ``items_per_thread`` features per sample. Each warp writes
one result per feature.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/examples/reduce_batched.py
   :language: python
   :start-after: # docs: start cutlass-reduce-batched
   :end-before: # docs: end cutlass-reduce-batched
   :dedent: 4

:ref:`Batched reduction <coop-cutlass-reduce-batched>` in the API reference
has the groups, operators and element types the call takes.

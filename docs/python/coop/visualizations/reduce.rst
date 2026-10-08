.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-reduce:

Reduce
======

:func:`cuda.coop.reduce` combines a group's values into one aggregate.
:func:`cuda.coop.sum` is the sum specialization. Blocks and warps accept one
scalar or several items per thread. Every input item contributes, and the
scalar result is defined only at group rank zero.

The explorer uses eight teaching threads. It shows four lanes per physical
warp and two per logical warp; **physical CUDA warps have 32 lanes**.

.. coop-visualization:: reduce

   .. only:: html

      .. figure:: reduce.svg
         :alt: Eight threads fold pairs of input values into partial sums,
               then reduce to 136 at thread zero; other return values
               are undefined.
         :width: 100%

         Block sum with two items per thread.

   Each thread first adds its own pair. The group combines those partials;
   only thread zero has a defined return value in this example.

   .. code-block:: text

      Thread          0     1     2     3     4      5      6      7
      Input          1,2   3,4   5,6   7,8  9,10  11,12  13,14  15,16
      Local sum       3     7    11    15    19     23     27     31
      Return value   136    ?     ?     ?     ?      ?      ?      ?

Reading the stages
------------------

The local fold produces one contribution per thread. The combine row shows
partial aggregates within each selected group. The final row shows the
aggregate at group rank zero. ``?`` marks return values that the program must
not read.

All reductions use CUB. A block can select ``raking_commutative_only``,
``raking``, or ``warp_reductions`` (the default). The first
requires a commutative operator; the explorer's sum and maximum satisfy that
requirement. Custom callbacks cannot select ``raking_commutative_only``
because the planner cannot establish their commutativity. Raking combines
contributions through shared-memory segments. Warp reductions combine warp
partials before producing the block result. These rows illustrate legal
combinations; they do not show exact instruction schedules. Floating-point
rounding can change when the combination order changes.

``valid_items`` counts contributing group members, starting at rank zero.
It is available for scalar block, physical-warp, and logical-warp reductions.
The count must be uniform within the group and
between one and the group size, inclusive. Every group member still
participates. The explorer offers this choice only for one item per thread.

A custom operator uses the qualified ``cuda.coop.numba_mlir`` namespace.
It must be associative and is supported through the CUB block or warp path,
with scalar or fixed-array inputs and a result defined only at the group
root. The common namespace accepts
built-in names such as ``"sum"``, ``"max"``, ``"min"``, ``"multiplies"``,
and the integer bitwise operators.

Using Reduce in a kernel
------------------------

This fragment runs inside a Numba-CUDA-MLIR kernel that accepts
``items_per_thread``. Import ``cuda`` from
``numba_cuda_mlir``, and import ``coop`` from ``cuda``. Launch with
128 threads and supply at least ``128 * items_per_thread`` input elements
for each block.

.. code-block:: python

   block = coop.this_block()
   values = coop.ThreadData(items_per_thread)
   coop.load(block, source, values, offset=cuda.blockIdx.x * 128 * items_per_thread)
   total = coop.sum(block, values, algorithm="raking")
   if cuda.threadIdx.x == 0:
       output[cuda.blockIdx.x] = total

The input ``values`` is unchanged. A logical warp uses
``coop.this_warp().group_by(8)``, yielding four groups of eight lanes inside
each physical warp. Each group has a separate aggregate, consumed by its
rank-zero lane. Logical widths may be powers of two from 1 through 32 or any
width from 17 through 31. CUB supports only one non-power-of-two group per
physical warp. For those widths, use ``group_by(width, exhaustive=False)`` and
guard the reduction with ``group.is_member()`` so trailing lanes do not participate:

.. code-block:: python

   group = coop.this_warp().group_by(8)
   total = coop.sum(group, source[cuda.threadIdx.x])
   if group.rank() == 0:
       output[cuda.threadIdx.x // 8] = total

For the explorer's custom-maximum choice, use a device callback and the
qualified API. Launch this kernel with one block whose size matches the
input:

.. code-block:: python

   from numba_cuda_mlir import cuda
   import cuda.coop.numba_mlir as coop

   @cuda.jit(device=True)
   def maximum(left, right):
       return left if left > right else right

   @cuda.jit
   def block_maximum(source, output):
       thread = cuda.threadIdx.x
       result = coop.reduce(
           coop.this_block(),
           source[thread],
           binary_op=maximum,
       )
       if thread == 0:
           output[0] = result

Block reductions accept ``temp_storage=coop.TempStorage(...)``. An explicit
descriptor defaults to caller-managed synchronization; use
``coop.TempStorage(auto_sync=True)`` for automatic synchronization before
reuse. Omitting ``temp_storage`` uses compiler-managed scratch with automatic
synchronization. Warp reductions always use compiler-managed scratch.

See :func:`cuda.coop.reduce` and the :doc:`../programming_guide` for the full
operation and group contracts.

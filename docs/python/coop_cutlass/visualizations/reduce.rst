.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-reduce:

Reduce
======

:func:`cuda.coop.cutlass.reduce` combines a group's values into one aggregate.
:func:`cuda.coop.cutlass.sum` is the sum specialization. Blocks and warps accept one
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
requirement. Raking combines
contributions through shared-memory segments. Warp reductions combine warp
partials before producing the block result. These rows illustrate legal
combinations; they do not show exact instruction schedules. Floating-point
rounding can change when the combination order changes.

``valid_items`` counts contributing group members, starting at rank zero.
It is available for scalar block, physical-warp, and logical-warp reductions.
The count must be uniform within the group and
between one and the group size, inclusive. Every group member still
participates. The explorer offers this choice only for one item per thread.

Built-in names include ``"sum"``, ``"max"``, ``"min"``, ``"multiplies"``,
and the integer bitwise operators. Recognized operator and NumPy aliases
select these same built-ins; custom callbacks are unsupported.

Using Reduce in a kernel
------------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads and supply at
least ``128 * items_per_thread`` input elements for each block.

.. code-block:: python

   block = coop.this_block()
   values = coop.load(
       block, source, items_per_thread=items_per_thread,
       offset=block_index * 128 * items_per_thread,
   )
   total = coop.sum(block, values, algorithm="raking")
   if thread_rank == 0:
       output[block_index] = total

The input ``values`` is unchanged. A logical warp uses
``coop.this_warp().group_by(8)``, yielding four groups of eight lanes inside
each physical warp. Each group has a separate aggregate, consumed by its
rank-zero lane. Logical widths may be powers of two from 1 through 32 or any
width from 17 through 31. CUB supports only one non-power-of-two group per
physical warp. For those widths, use ``group_by(width, exhaustive=False)`` and
guard the reduction with ``group.is_member()`` so trailing lanes do not participate.
This fragment writes one sum for each eight-lane logical warp:

.. code-block:: python

   group = coop.this_warp().group_by(8)
   total = coop.sum(group, source[thread_rank])
   if group.rank() == 0:
       output[thread_rank // 8] = total


Block reductions accept ``temp_storage=coop.TempStorage(...)``. An explicit
descriptor defaults to caller-managed synchronization; use
``coop.TempStorage(auto_sync=True)`` for automatic synchronization before
reuse. Omitting ``temp_storage`` uses compiler-managed scratch with automatic
synchronization. Warp reductions always use compiler-managed scratch.

Grid reduction is unsupported. See :func:`cuda.coop.cutlass.reduce` and
:ref:`the Programming Guide <coop-cutlass-reduce>` for groups and controls.

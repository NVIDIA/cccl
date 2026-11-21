.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-reduce:

Reduce
======

A reduction combines the values of a group's threads into one result, and
the thread with rank 0 in the group receives it.
:func:`~cuda.coop.cutlass.sum` adds the values, and
:func:`~cuda.coop.cutlass.reduce` takes the operator as an argument.

The explorer draws eight threads as one block, as two warps of four lanes,
or as four groups of two lanes.

.. coop-visualization:: reduce

   .. only:: html

      .. figure:: reduce.svg
         :alt: Eight threads each add their own two input values, and the
               group combines those sums into 136 in thread zero. The other
               return values are undefined.
         :width: 100%

         Block sum with two items per thread.

   Each thread first adds its own two items. The group combines those sums,
   and thread 0 receives the result.

   .. code-block:: text

      Thread          0     1     2     3     4      5      6      7
      Input          1,2   3,4   5,6   7,8  9,10  11,12  13,14  15,16
      Local sum       3     7    11    15    19     23     27     31
      Return value   136    ?     ?     ?     ?      ?      ?      ?

Reading the stages
------------------

The local row holds one value per thread, combined from that thread's own
items. The combine row shows partial results inside each group. The last row
shows the result in the thread with rank 0 of each group. ``?`` marks a
return value that the kernel must not read.

For a block, the combine row changes with the algorithm. On a block of more
than 32 threads, ``raking`` has every thread place its value in shared
memory, where one warp combines them. ``raking_commutative_only`` is a form
of ``raking`` for commutative operators. ``warp_reductions`` reduces inside
each warp and then combines the warps' results. The row is drawn for eight
threads and shows one valid grouping for each algorithm.

With one item per thread, the explorer can limit each group to its first
half with ``valid_items``. The other threads stay in the picture because
they still make the call. The explorer's operators are sum and maximum.
:ref:`Reduction <coop-cutlass-reduce>` in the API reference has the rules
for each control.

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

This fragment writes one sum for each group of eight lanes, sixteen for each
block. The first lane of each group holds its group's sum:

.. code-block:: python

   group = coop.this_warp().group_by(8)
   total = coop.sum(group, source[block_index * 128 + thread_rank])
   if group.rank() == 0:
       output[block_index * 16 + thread_rank // 8] = total

:ref:`Reduction <coop-cutlass-reduce>` in the API reference covers the
operators, ``valid_items``, groups of 17 to 31 lanes and ``temp_storage``.

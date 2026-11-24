.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-merge-sort:

Merge Sort
==========

A merge sort puts the keys held by a group's threads in order.
:func:`~cuda.coop.cutlass.merge_sort_keys` sorts the keys, and
:func:`~cuda.coop.cutlass.merge_sort_pairs` also carries a value with each
key, such as the position the key came from. The sorted keys go back to the
threads in :term:`blocked` order, and the inputs keep their values.

The explorer covers both functions. Choose the group, the number of items
per thread, the key order, distinct or repeated keys, and the number of
valid items. The eight teaching threads represent one block, two warps of
four lanes, or four groups of two lanes, which the explorer labels logical
warps.

.. coop-visualization:: merge-sort

   .. only:: html

      .. figure:: merge-sort.svg
         :alt: Eight threads hold two keys each, from 7 and 12 in thread 0
               to 13 and 2 in thread 7, and a letter from A to P marks each
               key. One row sorts each thread's two keys, and two rows
               merge the runs of two and of four threads. In the last row
               the keys run from 0 and 1 in thread 0 to 14 and 15 in thread
               7, each with its letter.
         :width: 100%

         ``merge_sort_pairs`` in ascending order over one block with two
         items per teaching thread. The letters A to P stand for the values
         0 to 15, each key's position in the tile.

   Each thread sorts a copy of its own items. Neighboring sorted runs are
   then merged, two at a time, until the group has one sorted sequence,
   which goes back to the threads in blocked order. The inputs keep their
   values.

   .. code-block:: text

      Thread          T0     T1    T2     T3     T4     T5     T6     T7
      Input keys     7,12   1,6   11,0   5,10   15,4   9,14   3,8   13,2
      Input letters   A,B   C,D    E,F    G,H    I,J    K,L    M,N    O,P
      Output keys     0,1   2,3    4,5    6,7    8,9   10,11  12,13  14,15
      Output letters  F,C   P,M    J,G    D,A    N,K    H,E    B,O    L,I

Reading the stages
------------------

The input row stays in view while copies of the items move through the
local sorts and the merges. A color marks the thread an item came from. In
pairs mode each key carries a letter, which the figure and the explorer
call a badge. The letter stands for the value that travels with the key,
which is the key's position in the input: ``A`` is 0, ``B`` is 1, and so on.
Select an item to see its value and where it is now. A value stays beside
its key through every stage.

In a kernel, :cpp:class:`cub::BlockMergeSort` sorts a block and
:cpp:class:`cub::WarpMergeSort` sorts a warp or a group of lanes. Choose
``Repeated keys`` to see equal keys meet. The explorer then takes the key
from the right-hand run first, which is one of the orders the sort may
return.

With ``Physical warp`` or ``Logical warp`` selected, the working row holds
one sorted run for each group, and every key stays inside its group.
Sorting every group therefore leaves the row as a whole unsorted, as
:ref:`this FAQ <coop-faq-global-sort>` explains for blocks.

Partial tiles
-------------

``Valid items per group`` sets ``valid_items``. The count runs across the
group's tile in blocked order, so it can end partway through a thread's
items. The explorer fades the inputs at or past the count. In the working
row and in the result, ``?`` marks a position that holds no sorted key. In
a kernel those positions of the result hold unspecified values. The
explorer's notes name an ``oob_default`` that suits its keys: a key larger
than every input for ascending order, and -1 for descending order.

The sort works on payloads. ``valid_items`` and ``oob_default`` have no
effect on what ``load`` reads or ``store`` writes, so pass ``valid_items``
to those calls too, as the kernel below does.
:ref:`Comparison sorting <coop-cutlass-merge-sort>` in the API reference
has the rules for both controls.

Using Merge Sort in a kernel
----------------------------

This kernel sorts a block's tile twice and leaves the last seven positions
out of both sorts. The first call sorts the keys in ascending order. The
second sorts them in descending order with a value beside each key. Both
calls share one ``TempStorage``. The last two ``store`` calls write ``keys``
and ``values``, which still hold what was loaded. ``_BLOCK`` is
``(8, 4, 2)``, a block of 64 threads.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/examples/merge_sort.py
   :language: python
   :start-after: # docs: start cutlass-merge-sort
   :end-before: # docs: end cutlass-merge-sort
   :dedent: 4

:ref:`Comparison sorting <coop-cutlass-merge-sort>` in the API reference
covers the groups, the element types and every control.

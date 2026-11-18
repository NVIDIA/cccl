.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-load:

Load
====

:func:`cuda.coop.cutlass.load` reads a group's :term:`tile` from memory and
returns each thread's items as a payload. The algorithm decides which thread
reads which element, and whether the threads then exchange their items
through shared scratch.

The explorer shows eight teaching threads that load a full tile. For the
two warp-transpose algorithms it draws them as two warps of four lanes.
**CUDA warps have 32 lanes**, and in a kernel these two algorithms need a
block size that is a multiple of 32.

.. coop-visualization:: load

   .. only:: html

      .. figure:: load-transpose.svg
         :alt: Transpose load: threads read striped values, exchange through shared scratch, and finish with consecutive values per thread.
         :width: 100%

         Transpose load with eight illustrative threads and two items per thread.

   Each column below belongs to one thread. Transpose first reads in striped
   order, then exchanges values to give each thread a consecutive pair.

   .. code-block:: text

      Thread          0     1     2     3     4     5     6     7
      Memory reads   0,8   1,9  2,10  3,11  4,12  5,13  6,14  7,15
      After exchange 0,1   2,3   4,5   6,7   8,9  10,11 12,13 14,15

Reading the stages
------------------

The memory row numbers the elements by their position in the tile. The next
row shows which thread reads each element, and the last row shows which
thread ends up with it. For an algorithm with an exchange, a scratch row
lies between the two. Select a value to see which thread reads it and
which thread and item receive it.

``direct`` and ``vectorize`` give a :term:`blocked` payload: thread ``t``
reads ``K`` consecutive elements starting at ``t * K``, where ``K`` is the
number of items per thread. For ``vectorize`` with more than one item per
thread, the explorer outlines each thread's elements in the memory row as
candidates for one vector access.

``striped`` gives thread ``t`` the elements ``t + i * G`` for item ``i``,
where ``G`` is the group's size. Neighboring threads read neighboring
addresses, and the payload stays :term:`striped`. ``transpose`` makes the
same reads and then exchanges the items into a blocked payload.

``warp_transpose`` reads in striped order inside each warp's part of the
tile and exchanges the items within that warp into a blocked payload.
``warp_transpose_timesliced`` makes the same reads, and the warps then take
turns with scratch sized for one warp. The explorer shows one exchange round
for each warp. Every warp has read its elements before the first round.

The :ref:`glossary <coop-glossary-layouts>` has a table of both layouts.
:ref:`Load & Store <memory-operations>` in the API reference has the rules
for each algorithm and control.

Using Load in a kernel
----------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads and provide at
least ``128 * items_per_thread`` source elements for each block.

.. code-block:: python

   block = coop.this_block()
   offset = block_index * 128 * items_per_thread
   items = coop.load(
       block, source, items_per_thread=items_per_thread,
       algorithm="transpose", offset=offset,
   )
   # Each thread holds items_per_thread consecutive elements of the tile.

:ref:`Load & Store <memory-operations>` in the API reference covers partial
tiles, offsets, and loads on warps and groups of lanes, with tested
examples.

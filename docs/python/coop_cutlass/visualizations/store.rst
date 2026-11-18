.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-store:

Store
=====

:func:`cuda.coop.cutlass.store` writes each thread's payload to a group's
:term:`tile` in memory. The algorithm names the layout of the payload, and
decides whether the threads exchange their items through shared scratch
before they write.

The explorer shows eight teaching threads that store a full tile. For the
two warp-transpose algorithms it draws them as two warps of four lanes. In
a kernel these two algorithms need a block size that is a multiple of 32.

.. coop-visualization:: store

   .. only:: html

      .. figure:: store-transpose.svg
         :alt: Transpose store exchanges blocked pairs into striped writer registers, then stores the values in consecutive memory locations.
         :width: 100%

         Transpose store with eight illustrative threads and two items per thread.

   Each thread begins with a consecutive pair. The exchange redistributes
   those values so adjacent threads can write adjacent memory locations.

   .. code-block:: text

      Thread          0     1     2     3     4     5     6     7
      Input          0,1   2,3   4,5   6,7   8,9  10,11 12,13 14,15
      Writer values  0,8   1,9  2,10  3,11  4,12  5,13  6,14  7,15
      Memory         0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

Reading the stages
------------------

The first row shows the payload each thread passes. The writer row shows
which thread writes each element, and the last row shows the tile in
memory. For an algorithm with an exchange, a scratch row lies between the
payload and the writers. Select a value to follow it from the thread that
passes it to the thread that writes it and to its place in the tile.

``direct`` and ``vectorize`` take a :term:`blocked` payload: thread ``t``
holds ``K`` consecutive elements starting at tile position ``t * K``, where
``K`` is the number of items per thread, and writes them itself.

``striped`` takes a :term:`striped` payload: thread ``t`` writes item ``i``
to tile position ``t + i * G``, where ``G`` is the group's size.
``transpose`` takes a blocked payload, exchanges the items into striped
order and then makes the same writes.

``warp_transpose`` makes that exchange within each warp, and each warp
writes its own part of the tile in striped order. With
``warp_transpose_timesliced`` the warps take turns with scratch sized for
one warp. The explorer shows one exchange round for each warp. All warps
write to memory after the last round.

The :ref:`glossary <coop-glossary-layouts>` has a table of both layouts.
:ref:`Load & Store <memory-operations>` in the API reference has the rules
for each algorithm and control, and says what a kernel may expect of the
payload after the call.

Using Store in a kernel
-----------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads. Give each block
at least ``128 * items_per_thread`` source and destination elements.

.. code-block:: python

   block = coop.this_block()
   offset = block_index * 128 * items_per_thread
   items = coop.load(
       block, source, items_per_thread=items_per_thread,
       algorithm="direct", offset=offset,
   )
   coop.store(block, destination, items, algorithm="transpose", offset=offset)
   # destination holds the block's tile in the order source had it.

:ref:`Load & Store <memory-operations>` in the API reference covers partial
tiles, offsets, and stores on warps and groups of lanes, with tested
examples.

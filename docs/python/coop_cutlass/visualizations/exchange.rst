.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-exchange:

Exchange
========

:func:`~cuda.coop.cutlass.exchange` changes which thread of a group holds
each item of a :term:`tile`. The explorer steps through its eight modes on
eight threads with one, two or four items each.

.. coop-visualization:: exchange

   .. only:: html

      .. figure:: exchange-striped.svg
         :alt: Eight threads hold two items each. Thread 0 holds 0 and 8
               before the exchange and 0 and 1 after it. Thread 7 holds 7
               and 15 before it and 14 and 15 after it.
         :width: 100%

         ``striped_to_blocked`` on eight threads with two items each.

   Each value keeps its position in the tile. Every value except 0 and 15
   moves to another thread.

   .. code-block:: text

      Thread          0     1     2     3     4     5     6     7
      Striped input  0,8   1,9  2,10  3,11  4,12  5,13  6,14  7,15
      Blocked result 0,1   2,3   4,5   6,7   8,9  10,11 12,13 14,15

Reading the stages
------------------

The first row groups the input by thread. The second row is the shared
scratch, in tile order. The third row groups the result by thread. Colors
show the thread a value came from. Select a value to see its slot in the
input and in the result.

``blocked_to_warp_striped`` stripes the items inside each warp, and
``warp_striped_to_blocked`` turns that layout back into a blocked one. The
explorer draws warps of four lanes so that two fit on screen. With two
warps and two or four items per thread, the warp-striped layout differs from
the striped layout, which is striped across the whole block.

For the scatter modes the explorer starts from a blocked input and uses
``rank[p] = (5 * p) % N`` on a tile of ``N`` items, which sends every item
to a position of its own. For ``scatter_to_striped_guarded`` it gives every
fifth item the rank -1, and for ``scatter_to_striped_flagged`` it gives
every fourth item a zero flag. A fourth row then holds the items that were
skipped, and ``?`` marks each result that no item was sent to. That result
is undefined.

The explorer draws the exchange without ``warp_time_slicing``.

Using Exchange in a kernel
--------------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads and provide at
least ``128 * items_per_thread`` source and destination elements for each
block. The fragment loads a striped payload, makes it blocked and stores it.

.. code-block:: python

   block = coop.this_block()
   offset = block_index * 128 * items_per_thread
   items = coop.load(
       block, source, items_per_thread=items_per_thread,
       algorithm="striped", offset=offset,
   )
   blocked = coop.exchange(block, items, mode="striped_to_blocked")
   coop.store(block, destination, blocked, algorithm="direct", offset=offset)
   # items is still striped. blocked holds the same tile in blocked layout.

This tested kernel reverses a block's tile with ``scatter_to_blocked``.
Each item's rank is the last tile position minus its own position.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_collective_documentation.py
   :language: python
   :start-after: # cutlass-scatter-example-begin
   :end-before: # cutlass-scatter-example-end
   :dedent: 4

:ref:`Data rearrangement <coop-cutlass-exchange>` in the API reference has
the rules for each mode, for ``ranks`` and ``valid_flags``, and for
``warp_time_slicing``. The :ref:`glossary <coop-glossary-layouts>` has a
table of the blocked and striped layouts.

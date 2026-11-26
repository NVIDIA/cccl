.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-shuffle:

Shuffle
=======

:func:`~cuda.coop.cutlass.shuffle` shifts a block's values along the order
of its threads. With a payload it shifts the block's items by one position.
With one scalar per thread it gives each thread the scalar of the thread
``distance`` ranks ahead of it.

The explorer shows the four modes on a block of eight threads. For ``up``
and ``down``, each thread holds one, two or four items. ``offset`` and
``rotate`` move one scalar per thread, with the same distance in every
thread.

.. coop-visualization:: shuffle

   .. only:: html

      .. figure:: shuffle-up.svg
         :alt: Eight threads hold the values 0 to 15, two each. In the row
               below, the values 0 to 14 sit one position to the right, and
               the first position holds a question mark.
         :width: 100%

         ``up`` on eight threads with two items each.

   ``up`` gives each position the item before it. ``?`` marks the result
   with no source, which is undefined.

   .. code-block:: text

      Thread     0     1     2     3     4     5      6      7
      Input     0,1   2,3   4,5   6,7   8,9  10,11  12,13  14,15
      Up        ?,0   1,2   3,4   5,6   7,8   9,10  11,12  13,14
      Down      1,2   3,4   5,6   7,8  9,10  11,12  13,14   15,?

Reading the stages
------------------

The first row holds the input. For ``up`` and ``down``, each thread puts
one item into shared scratch for its neighbor: its last item for ``up`` and
its first item for ``down``. Its other items move between its own slots.
For ``offset`` and ``rotate``, every thread puts its scalar into scratch.
The last row holds the result, grouped by thread.

Select a value to follow it from its source to its destination. The shared
row shows the values that pass between threads.

Using Shuffle in a kernel
-------------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads and provide at
least ``128 * items_per_thread`` source and destination elements for each
block.

.. code-block:: python

   block = coop.this_block()
   offset = block_index * 128 * items_per_thread
   items = coop.load(
       block, source, items_per_thread=items_per_thread, offset=offset
   )
   shifted = coop.shuffle(block, items, mode="up")
   for item in cutlass.range_constexpr(items_per_thread):
       position = thread_rank * items_per_thread + item
       if position > 0:
           destination[offset + position] = shifted[item]

Every thread makes the call. The ``if`` guards the write that follows it,
so the undefined first item is never read. The first element of each
block's tile in ``destination`` keeps the value it had.

This tested kernel rotates one scalar per thread by seven ranks.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_collective_documentation.py
   :language: python
   :start-after: # cutlass-rotate-example-begin
   :end-before: # cutlass-rotate-example-end
   :dedent: 4

:ref:`Data rearrangement <coop-cutlass-shuffle>` in the API reference has
the rules for each mode and for ``distance``.

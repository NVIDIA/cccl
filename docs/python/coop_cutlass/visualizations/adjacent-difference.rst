.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-adjacent-difference:

Adjacent Difference
===================

:func:`cuda.coop.cutlass.adjacent_difference` subtracts a neighboring item from
each item of a block's tile. The neighbor may belong to another thread.

In the explorer, choose left or right to follow that dependency, then change
``valid_items`` or supply the item on the other side of the tile's edge. The
explorer uses eight teaching threads and small ``Int32`` values, so every
subtraction fits in the element type.

.. coop-visualization:: adjacent-difference

   .. only:: html

      .. figure:: adjacent-difference.svg
         :alt: With five valid inputs and predecessor zero, left differences
               of 2, 2, 5, 5, 9, 9, 9, 3 are 2, 0, 3, 0, 4, 9, 9, 3. The last
               three items are copied unchanged.
         :width: 100%

         Left differences with ``valid_items=5`` and
         ``tile_predecessor_item=0``. The three items at or past the count
         are copied from the input.

   For ``direction="left", valid_items=5, tile_predecessor_item=0``:

   .. code-block:: text

      Position       0  1  2  3  4 | 5  6  7
      Input          2  2  5  5  9 | 9  9  3
      Left neighbor  0  2  2  5  5 | not used
      Result         2  0  3  0  4 | 9  9  3
      Input after    2  2  5  5  9 | 9  9  3

Reading the stages
------------------

Positions follow :ref:`blocked order <coop-glossary-layouts>`: all of
thread 0's items, then thread 1's, and so on. With ``K`` items per thread,
item ``i`` of thread ``t`` has position ``t * K + i``. A left difference
computes ``input[p] - input[p - 1]``, and a right difference computes
``input[p] - input[p + 1]``.

The input row stays in place. The neighbor row shows the value each
position reads, including reads across thread boundaries. The final row is
the new payload. Select any result to see its subtraction or the reason it
copies the input.

At the edge of the tile, an item with no neighbor is copied.
``tile_predecessor_item`` supplies the missing left neighbor, and
``tile_successor_item`` supplies the missing right neighbor.

With ``valid_items``, the call treats the tile as ending at the count. The
positions at or past the count read no neighbor and copy their input. A
right difference with ``valid_items`` takes no ``tile_successor_item``, so
the explorer's "Tile neighbor value" control offers only "Not supplied" for
that combination. :ref:`Neighbor comparisons <coop-cutlass-neighbors>` in
the API reference has the rules for each control.

Using Adjacent Difference in a kernel
-------------------------------------

This tested kernel computes the left differences and both sets of flags of
one tile of a block of 128 threads. The two calls share one ``TempStorage``
created with ``auto_sync=True``, so a barrier follows each of them. No
``tile_predecessor_item`` is given, so the first difference is a copy of the
first item.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_neighbors.py
   :language: python
   :start-after: docs: start cutlass-neighbors
   :end-before: docs: end cutlass-neighbors
   :dedent: 4

For integer items, an inclusive sum of these differences gives back the
input, because the first difference is the first item.

To continue a sequence across tiles, pass the last item of the previous tile
as ``tile_predecessor_item``. Read that item from the input array, and write
the differences to a separate array, so that the input keeps its values for
a later tile or another block to read. The second example of
:func:`cuda.coop.cutlass.adjacent_difference` does this in a loop over tiles.

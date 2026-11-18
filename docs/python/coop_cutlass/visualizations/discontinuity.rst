.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-discontinuity:

Discontinuity
=============

:func:`cuda.coop.cutlass.discontinuity` flags the items of a block's tile that
differ from their neighbor. A head flag marks the first item of a run of
equal values, and a tail flag marks the last.

Runs can cross thread boundaries. The explorer shows eight teaching
threads, with separate rows for the neighbor values and the returned flags.
Supply a tile predecessor or successor to see whether a run continues
outside the tile.

.. coop-visualization:: discontinuity

   .. only:: html

      .. figure:: discontinuity.svg
         :alt: Input 2, 2, 5, 5, 9, 9, 9, 3 produces heads 1, 0, 1, 0, 1, 0,
               0, 1 and tails 0, 1, 0, 1, 0, 0, 1, 1. The first head and last
               tail are one when no outside neighbors are supplied.
         :width: 100%

         Head and tail flags of one tile with no tile predecessor or
         successor. The run of three nines spans three threads here.

   With no tile predecessor or successor:

   .. code-block:: text

      Input        2  2  5  5  9  9  9  3
      Heads        1  0  1  0  1  0  0  1
      Tails        0  1  0  1  0  0  1  1
      Input after  2  2  5  5  9  9  9  3

Reading the stages
------------------

Positions follow :ref:`blocked order <coop-glossary-layouts>`: all of
thread 0's items, then thread 1's, and so on. A head flag tests
``previous != current``. A tail flag tests ``current != next``. A boundary
between threads is an ordinary adjacent pair.

When no predecessor is supplied, the first head is 1. When no successor is
supplied, the last tail is 1. A supplied value replaces that flag with the
comparison. The explorer's first item is 2, so setting
``tile_predecessor_item`` to 2 changes the first head from 1 to 0 and leaves
every other flag as it was.

:ref:`Neighbor comparisons <coop-cutlass-neighbors>` in the API reference
has the rules for each control.

Using Discontinuity in a kernel
-------------------------------

This tested kernel computes both sets of flags and the left differences of
one tile of a block of 128 threads. ``mode="heads_and_tails"`` returns the
head flags and the tail flags from one call. The two calls share one
``TempStorage`` created with ``auto_sync=True``, so a barrier follows each
of them.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_neighbors.py
   :language: python
   :start-after: docs: start cutlass-neighbors
   :end-before: docs: end cutlass-neighbors
   :dedent: 4

An inclusive sum of the head flags, minus one, numbers the runs of this tile
from zero. The numbers start again in each tile. Numbering runs across
tiles takes another step that counts the runs of the earlier tiles and joins
a run that spans two tiles.

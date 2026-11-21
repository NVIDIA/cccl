.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Adjacent difference and discontinuity
=====================================

Adjacent difference subtracts neighboring values. Discontinuity marks the
start or end of each run of equal values. A run boundary is local to the input
Tile, even when an adjacent Tile contains the same value.

Interface and examples
----------------------

These calls compute both differences and both boundary masks for the same Tile.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto left = coop::adjacent_difference_subtract_left(input);
   auto right = coop::adjacent_difference_subtract_right(input);
   auto [heads, tails] = coop::discontinuity_flag_heads_and_tails(input);

Both adjacent-difference functions preserve the input numeric element type
and positive extent ``N``. Left subtracts the preceding element and preserves
the first input; right subtracts the next element and preserves the last.
Discontinuity returns ``int`` masks of extent ``N``. Heads mark the first item
and unequal predecessors; tails mark the last item and unequal successors.
The paired function returns heads followed by tails.

The C++ functions have the same names and accept optional
``adjacent_difference_policy<>`` or ``discontinuity_policy<>`` objects.
Both policies use the default implementation.

Behavior
--------

Subtraction and inequality are fixed operations. These functions do not accept
custom operators, predecessor values, or successor values. Signed and unsigned
arithmetic follow the input dtype; choose a wider input type before subtracting
if the mathematical difference can exceed its range. Floating-point
inequality follows the underlying numeric comparison, so NaNs flag boundaries.

For input ``[2, 2, 5, 5, 5, 1, 1, 4]``:

.. code-block:: text

   heads:             [1, 0, 1, 0, 0, 1, 0, 1]
   tails:             [0, 1, 0, 0, 1, 0, 1, 1]
   subtract_left:     [2, 0, 3, 0, 0, -4, 0, 3]
   subtract_right:    [0, -3, 0, 0, 4, 0, -3, 4]

The preserved end values in adjacent difference differ from padding a missing
neighbor with the same value, which would produce zero at that boundary.

.. figure:: /tile/visualizations/adjacent-discontinuity.svg
   :alt: Neighbor subtraction, head and tail masks, and composed run IDs.

   Neighbor subtraction, head and tail masks, and composed run IDs.

:ref:`Neighbor comparison declarations and overloads <tile-cpp-adjacent-discontinuity-declarations>`.

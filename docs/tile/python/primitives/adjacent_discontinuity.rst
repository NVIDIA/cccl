.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Adjacent difference and discontinuity
=====================================

.. py:currentmodule:: cuda.coop.tile

Adjacent difference subtracts neighboring values. Discontinuity marks the
start or end of each run of equal values. A run boundary is local to the input
Tile, even when an adjacent Tile contains the same value.

Interface and examples
----------------------

.. py:function:: adjacent_difference_subtract_left(value, /, *, methods=None)
   :no-contents-entry:

   Return ``value[i] - value[i - 1]``; preserve ``value[0]``.

.. py:function:: adjacent_difference_subtract_right(value, /, *, methods=None)
   :no-contents-entry:

   Return ``value[i] - value[i + 1]``; preserve ``value[N - 1]``.

.. py:function:: discontinuity_flag_heads(value, /, *, methods=None)
   :no-contents-entry:

   Return 1 at the first item and where the value differs from its predecessor.

.. py:function:: discontinuity_flag_tails(value, /, *, methods=None)
   :no-contents-entry:

   Return 1 at the last item and where the value differs from its successor.

.. py:function:: discontinuity_flag_heads_and_tails(value, /, *, methods=None)
   :no-contents-entry:

   Return the tuple ``(heads, tails)`` with both discontinuity masks.

.. rubric:: Parameters and results

``value`` is a numeric Tile of shape ``(N,)``. Adjacent difference returns the
same dtype and shape. Discontinuity returns ``int32`` masks of shape ``(N,)``;
each element is 0 or 1. The combined function returns two such Tiles in head,
then tail order. ``methods`` must be omitted or empty.

.. rubric:: Example: number equal-value runs

An inclusive sum of the head mask gives one-based run IDs. This example has
64 runs of four equal integers each.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def run_ids_kernel(source, destination):
       values = ct.load(source, (ct.bid(0),), shape=(256,))
       heads = coop.discontinuity_flag_heads(values)
       run_ids = coop.inclusive_sum(heads)
       ct.store(destination, (ct.bid(0),), run_ids)

   source = torch.arange(256, device="cuda", dtype=torch.int32) // 4
   destination = torch.empty_like(source)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), run_ids_kernel, (source, destination))
   assert torch.equal(destination, source + 1)


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

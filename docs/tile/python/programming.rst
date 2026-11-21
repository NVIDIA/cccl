.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Programming guide
=================

Prefix sums within a tile
-------------------------

Suppose each element is a count. To pack that many items per element, you
need the starting offset for each one. For counts beginning with
``[2, 3, 5, 7]``, the offsets start at ``[0, 2, 5, 10]``.
This kernel computes those offsets independently in each 256-element tile.

.. literalinclude:: ../../../python/cuda_coop_tile/examples/block_scan.py
   :language: python
   :start-at: import cuda.tile as ct
   :end-before: def launch_block_exclusive_sum

``ct.bid(0)`` gives the logical block index. The shaped ``ct.load`` takes a
**tile coordinate**. Block 0 reads elements 0 through 255 and block 1 reads
elements 256 through 511. Multiplying the index by 256 here would skip tiles.

``coop.exclusive_sum(values)`` returns a tile of the same shape and dtype.
Each element holds the sum of the elements before it in this tile. The
first offset in *each* tile is zero. ``ct.store`` writes those offsets to the
matching output partition.

Run and check both blocks
-------------------------

After :doc:`installation`, download :download:`block_scan.py
<../../../python/cuda_coop_tile/examples/block_scan.py>` and run it in that
directory with PyTorch and a compatible GPU:

.. code-block:: console

   $ python block_scan.py
   [[0, 0, 1, 3], [0, 256, 513, 771]]

The first input tile starts with ``[0, 1, 2, 3]`` and the second with
``[256, 257, 258, 259]``. The program checks every offset against a separate
cumulative sum for each row before printing the first four from each tile.
Here is its launch and check:

.. literalinclude:: ../../../python/cuda_coop_tile/examples/block_scan.py
   :language: python
   :start-at: def launch_block_exclusive_sum

To process more complete tiles, use matching contiguous one-dimensional
device arrays and launch one block per 256 elements. Skip the launch for an
empty input. See the :doc:`shared programming model </tile/programming_model>`
for partial tiles.

Choose the result you need
--------------------------

**A prefix for each element.** Set a starting offset with ``prefix_value``.
For other operations, choose inclusive or exclusive scan. These calls use a
loaded integer tile, with ``block_prefix`` supplied by the caller.

.. code-block:: python

   exclusive = coop.exclusive_sum(values, prefix_value=block_prefix)
   maximum = coop.inclusive_scan(values, scan_op="max")
   product = coop.exclusive_scan(values, scan_op="*", initial_value=1)

The :doc:`scan reference <primitives/scan>` describes seeds, operation
names and algorithms.

**One result per row.** On an ``(8, 128)`` matrix,
``coop.sum(matrix, axis=1)`` returns eight sums in a tile of shape ``(8,)``.
Use ``keepdims=True`` to keep shape ``(8, 1)`` for row-wise broadcasting.
For example,
``matrix - coop.reduce(matrix, axis=1, keepdims=True, binary_op="min")``
subtracts each row's minimum from that row. See the
:doc:`reduction reference <primitives/reduce>` for the supported shapes.

**Selected keys and their original positions.** Pass a tile of column indices
alongside the keys to :py:func:`cuda.coop.tile.topk_max_pairs`. Read the
selected prefix from each output tile. The :ref:`MMA and TopK example
<tile-python-mma-topk>` uses the indices to identify which matrix columns
produced the highest scores.

.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-adjacent-difference:

Adjacent Difference
===================

:func:`cuda.coop.adjacent_difference` subtracts a neighboring item from
each item in a block tile. The result has the input's dtype, extent, and
blocked ownership. The input remains unchanged.

The neighbor may belong to another thread. Choose left or right to follow
that dependency, then change ``valid_items`` or supply a value just outside
the tile. The explorer uses eight teaching threads and small ``int32``
values; every subtraction fits in that dtype.

.. coop-visualization:: adjacent-difference

   .. only:: html

      .. figure:: adjacent-difference.svg
         :alt: With five valid inputs and predecessor zero, left differences
               of 2, 2, 5, 5, 9, 9, 9, 3 are 2, 0, 3, 0, 4, 9, 9, 3. The last
               three items are copied unchanged.
         :width: 100%

         Left differences for five valid items. The invalid suffix is copied
         from the input into the returned payload.

   For ``direction="left", valid_items=5, tile_predecessor_item=0``:

   .. code-block:: text

      Position       0  1  2  3  4 | 5  6  7
      Input          2  2  5  5  9 | 9  9  3
      Left neighbor  0  2  2  5  5 | not used
      Result         2  0  3  0  4 | 9  9  3
      Input after    2  2  5  5  9 | 9  9  3

Reading the stages
------------------

Flatten the :ref:`blocked payloads <coop-glossary-layouts>` in thread-rank
order. With ``K`` items per thread, slot ``i`` of thread ``t`` has position
``t * K + i``. Left Difference computes ``input[p] - input[p - 1]``;
Right Difference computes ``input[p] - input[p + 1]``. Both subtract the
neighbor from the current item.

The input row stays in place. The neighbor row shows the value each
position reads, including reads across thread boundaries. The final row
contains a new payload. Select any result to see its subtraction or the
reason it copies the input.

At the tile edge, an omitted neighbor leaves the current input unchanged.
``tile_predecessor_item`` supplies the missing left neighbor;
``tile_successor_item`` supplies the missing right neighbor. Neither
argument wraps around to the other end of the tile. These scalar boundary
values must match the input dtype and agree across the block.

Every thread in the complete block must call the operation, including for
partial tiles; warp groups are unsupported. The function infers dtype and
per-thread extent from ``ThreadData``. It accepts automatic scratch or an
explicit :ref:`TempStorage <coop-temp-storage>`.

For a partial tile, ``valid_items`` must agree across the block. Only
positions below that count compute a difference. Every later position
copies its input, so those slots must also be initialized. A count of zero
copies the entire tile. A count equal to the tile capacity processes the
entire tile.

Right partial tiles do not accept ``tile_successor_item``, even when the
supplied count equals the capacity. The last valid item is copied
unchanged. The explorer disables the boundary-value control for this
combination, matching the API's rejection of that CUB overload.

Encoding deltas across block tiles
----------------------------------

With four items per thread, this kernel uses 128 threads to process each
full 512-item tile. The previous tile's final source item supplies
the left boundary.
The first tile uses zero so its first result retains the first input.
An inclusive sum of the resulting deltas reconstructs the original sequence
when the first delta is the first original value.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_neighbors.py
   :language: python
   :start-after: # adjacent-difference-example-begin
   :end-before: # adjacent-difference-example-end
   :dedent: 4

Use distinct global input and output buffers: an in-place multiblock launch
could overwrite a predecessor before the following block reads it. The
input-preservation guarantee for per-thread payloads does not remove that
global-memory race.

See :ref:`the Numba-CUDA-MLIR programming guide <coop-neighbor-comparisons>`
and :func:`cuda.coop.numba_mlir.adjacent_difference` for the qualified
``difference_op`` argument and local-array support.

.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-neighbor-operations:

Neighbor operations
===================

Adjacent Difference and Discontinuity read neighboring items in a block's
flattened, blocked sequence. They return new per-thread payloads and
preserve their inputs. A neighbor can belong to the same thread, another
thread, or the preceding or following tile when you provide its boundary
value.

Use :func:`cuda.coop.adjacent_difference` to compute changes between
numeric samples. Its default operation subtracts the neighbor from the
current item. Left differences can encode a sequence of values as deltas;
an inclusive sum reconstructs the sequence when the first delta is the
first original value. The
:doc:`Adjacent Difference explorer <visualizations/adjacent-difference>`
includes a tested kernel that supplies predecessors across block tiles.

Use :func:`cuda.coop.discontinuity` to find changes in keys or labels.
Head flags identify run starts, and tail flags identify run ends. An
inclusive sum of head flags assigns a run number to each item within the
tile. The :doc:`Discontinuity explorer <visualizations/discontinuity>`
shows both flags and includes a tested composition with Scan.

Tile boundaries and partial inputs
----------------------------------

A thread boundary does not end the sequence. Only the block tile's outer
edges need special treatment. For a left difference or a head flag, supply
``tile_predecessor_item`` to continue the sequence from an earlier tile.
For a full-tile right difference or a tail flag, supply
``tile_successor_item`` to continue into the next tile. These scalar values
must match the input dtype and agree across the block.

Adjacent Difference accepts a block-uniform ``valid_items`` count.
Positions outside the valid prefix copy their original input, so initialize
the entire payload. Right partial tiles cannot use a successor value; the
last valid item stays unchanged. Discontinuity requires a full tile, and
arbitrary padding can alter the final valid tail flag.

Every member of the complete block must call the operation. Warp groups
are unsupported. The functions infer dtype and per-thread extent from
``ThreadData``. See :doc:`programming_guide` for group participation and
:ref:`temporary storage <coop-temp-storage>` for caller-owned scratch.
Both operations can use automatic scratch or an explicit ``TempStorage``.

Custom operations with Numba-CUDA-MLIR
--------------------------------------

The qualified functions in ``cuda.coop.numba_mlir`` also accept fixed-size
local arrays and stateless device-compilable binary callables.
``difference_op(current, neighbor)`` returns the input dtype. Its argument
order stays the same for left and right differences.

For Discontinuity, ``flag_op(previous, current)`` determines heads and
``flag_op(current, next)`` determines tails. The distinction matters for a
predicate such as ``<``. Without a supplied tile boundary, the first head
or last tail remains one regardless of the predicate. Returned flags have
``int32`` dtype and the same per-thread extent as the input.

The common and qualified references are
:func:`cuda.coop.adjacent_difference`, :func:`cuda.coop.discontinuity`,
:func:`cuda.coop.numba_mlir.adjacent_difference`, and
:func:`cuda.coop.numba_mlir.discontinuity`.

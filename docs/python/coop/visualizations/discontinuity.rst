.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-discontinuity:

Discontinuity
=============

Both Numba-CUDA-MLIR and CUTLASS implement Discontinuity with the built-in
inequality predicate. The :ref:`CuTe example <coop-cutlass-neighbors>`
computes both head and tail flags while preserving its input.

:func:`cuda.coop.discontinuity` flags changes between adjacent values in a
full block tile. A head marks the start of a run; a tail marks its end.
The default predicate compares neighboring values for inequality.
``mode="heads_and_tails"`` returns both ``int32`` payloads as
``(heads, tails)``. Each flag stays in its input item's blocked slot, and
the input remains unchanged.

Runs can cross thread boundaries. The explorer shows eight teaching
threads, with separate rows for the neighbor values and returned flags.
Supply a tile predecessor or successor to see whether a run continues
outside the tile.

.. coop-visualization:: discontinuity

   .. only:: html

      .. figure:: discontinuity.svg
         :alt: Input 2, 2, 5, 5, 9, 9, 9, 3 produces heads 1, 0, 1, 0, 1, 0,
               0, 1 and tails 0, 1, 0, 1, 0, 0, 1, 1. The first head and last
               tail are one when no outside neighbors are supplied.
         :width: 100%

         Head and tail flags for a full tile with no supplied boundary
         values. The run of three nines spans three threads here.

   With no tile predecessor or successor:

   .. code-block:: text

      Input        2  2  5  5  9  9  9  3
      Heads        1  0  1  0  1  0  0  1
      Tails        0  1  0  1  0  0  1  1
      Input after  2  2  5  5  9  9  9  3

Reading the stages
------------------

Input positions follow :ref:`blocked order <coop-glossary-layouts>`:
all of thread zero's items, then thread one's, and so on. Head flags test
``previous != current``. Tail flags test ``current != next``. A boundary
between threads is an ordinary adjacent pair.

When no predecessor is supplied, the first head is one. When no successor
is supplied, the last tail is one. Supplying the appropriate tile boundary
value replaces that forced flag with the predicate result. For example,
``tile_predecessor_item=2`` changes the first head in the fallback table
from one to zero. It does not change any other flag. Scalar boundary values
must match the input dtype and agree across the block.

Every thread in the complete block must call the operation; warp groups
are unsupported. The function infers dtype and per-thread extent from
``ThreadData``. It accepts automatic scratch or an explicit
:ref:`TempStorage <coop-common-storage>`.

The operation requires a full tile. There is no ``valid_items`` argument.
Padding becomes input: a padding value equal to the final valid value can
make its tail zero, while a different padding value can make it one.
Masking stores afterward does not repair that comparison. Use this API
only when the entire tile and its boundary values have the intended
meaning.

Turning heads into run IDs
--------------------------

An inclusive sum of head flags counts how many runs have started. Subtract
one to obtain zero-based IDs. With four items per thread, this kernel uses
128 threads to process each full 512-item tile.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_neighbors.py
   :language: python
   :start-after: # discontinuity-example-begin
   :end-before: # discontinuity-example-end
   :dedent: 4

The IDs restart at zero in each block. Global run IDs require a separate
step to account for runs in preceding tiles and runs spanning tile
boundaries.

See :ref:`the Numba-CUDA-MLIR programming guide <coop-neighbor-comparisons>`
and :func:`cuda.coop.numba_mlir.discontinuity` for the qualified binary
``flag_op`` predicate and local-array support.

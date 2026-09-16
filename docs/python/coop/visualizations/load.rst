.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-load:

Load
====

:func:`cuda.coop.load` fills each thread's ``ThreadData`` from a contiguous
tile in memory. The algorithm determines which thread reads each value and
whether a shared-memory exchange rearranges it afterward.

The explorer shows eight illustrative threads. For the warp-transpose
algorithms, it groups them into two four-lane teaching warps. **CUDA warps
have 32 lanes**; executable blocks using these algorithms must contain a
multiple of 32 threads. All illustrated loads read a full tile.

.. coop-visualization:: load

   .. only:: html

      .. figure:: load-transpose.svg
         :alt: Transpose load: threads read striped values, exchange through shared scratch, and finish with consecutive values per thread.
         :width: 100%

         Transpose load with eight illustrative threads and two items per thread.

   Each column below belongs to one thread. Transpose first reads in striped
   order, then exchanges values to give each thread a consecutive pair.

   .. code-block:: text

      Thread          0     1     2     3     4     5     6     7
      Memory reads   0,8   1,9  2,10  3,11  4,12  5,13  6,14  7,15
      After exchange 0,1   2,3   4,5   6,7   8,9  10,11 12,13 14,15

Reading the stages
------------------

The memory row numbers values by their position in the tile. The next row
shows the threads that read them. For algorithms with an exchange, shared
scratch connects those readers to the final owners. Select a value to
inspect its source index and destination thread slot.

``direct`` and ``vectorize`` produce blocked ownership: thread ``t`` receives
``K`` consecutive values starting at ``t * K``, where ``K`` is the number of
items per thread. Vectorization can combine a thread's adjacent values into
a wider access when the pointer, alignment, type, and item count permit it.
The diagram does not guarantee a vector instruction.

``striped`` gives thread ``t`` the values at ``t + j * T`` for item slot
``j`` and group size ``T``. Neighboring threads read neighboring addresses
at each slot, and the output stays striped. ``transpose`` starts with those
same reads and exchanges the values into blocked ownership.

``warp_transpose`` reads a separate striped tile within each physical warp
and exchanges it into blocked ownership. ``warp_transpose_timesliced`` uses
the same memory reads, then lets warps take turns through a shared scratch
region sized for one warp. The animation separates those exchange rounds;
it does not serialize the global-memory loads.

Using Load in a kernel
----------------------

This fragment uses the common API inside a Numba-CUDA-MLIR kernel, with
``cuda`` imported from ``numba_cuda_mlir``, ``numpy as np``, and
``cuda.coop as coop``. Launch with 128 threads and provide at least 256
source elements for each block.

.. code-block:: python

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   offset = cuda.blockIdx.x * 256
   coop.load(block, source, items, algorithm="transpose", offset=offset)
   # Each thread now owns two consecutive values. Load returns None.

For a partial final tile, supply ``valid_items`` and, when needed,
``oob_default``. See :func:`cuda.coop.load` for the full parameter contract
and a complete example, and the :doc:`../programming_guide` for backend
activation and thread groups.

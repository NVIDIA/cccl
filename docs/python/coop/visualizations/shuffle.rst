.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-shuffle:

Shuffle
=======

:func:`cuda.coop.shuffle` shifts values within a complete block and returns a
new per-thread payload. The input remains unchanged. ``up`` and ``down`` move
one item along the flattened blocked tile, including across thread boundaries.
The qualified Numba-CUDA-MLIR API also supports scalar ``offset`` and ``rotate``.

The explorer shows eight illustrative threads. Change the items per thread
to see how local shifts connect across threads. Scalar modes use one value
per thread. All block members must participate in each collective.

.. coop-visualization:: shuffle

   .. only:: html

      .. figure:: shuffle-up.svg
         :alt: Up shuffle moves each input one position to the right in blocked order; the first output has no source and is undefined.
         :width: 100%

         Up with eight illustrative threads and two items per thread.

   A unit Up shift gives each position its predecessor. ``?`` marks an
   undefined output, which must not be read.

   .. code-block:: text

      Thread     0     1     2     3     4     5      6      7
      Input     0,1   2,3   4,5   6,7   8,9  10,11  12,13  14,15
      Up        ?,0   1,2   3,4   5,6   7,8   9,10  11,12  13,14
      Down      1,2   3,4   5,6   7,8  9,10  11,12  13,14   15,?

Reading the stages
------------------

The first row contains working copies of the input. For Up and Down, each
thread makes its boundary item available through shared scratch. The other
items shift between that thread's own slots. The last row places the resulting
values in blocked ownership.

``up`` reads ``input[p - 1]`` for output position ``p``; ``down`` reads
``input[p + 1]``. They support only the compile-time distance ``1``. The
first Up output and last Down output are undefined. Neither mode wraps or
copies the original boundary value into the result.

The qualified scalar modes use one scalar per thread. ``offset`` reads
thread ``t + distance``, with an undefined result when that thread lies
outside the block. ``rotate`` reads ``(t + distance) % block_size`` and
requires ``1 <= distance < block_size``. Both accept runtime distances,
which may differ between threads; the explorer uses a uniform distance.

Select a value to follow its source and destination. The shared row shows
the values communicated between threads, not a bank layout or a count of
hardware instructions.

Using Shuffle in a kernel
-------------------------

This fragment uses the common API inside a Numba-CUDA-MLIR kernel, with
``cuda`` imported from ``numba_cuda_mlir``, ``numpy as np``, and
``cuda.coop as coop``. Launch with 128 threads and provide at least 256
source and destination elements per block.

.. code-block:: python

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   offset = cuda.blockIdx.x * 256
   coop.load(block, source, items, offset=offset)
   shifted = coop.shuffle(block, items, mode="up")
   for item in range(2):
       position = cuda.threadIdx.x * 2 + item
       if position > 0:
           destination[offset + position] = shifted[item]

Every thread calls Shuffle. The conditional guards only the write afterward,
so the undefined first result is never consumed. The first element of each
block's destination tile is left unchanged; initialize those elements
separately if they need a value.

For a wrapped scalar neighbor, import ``cuda.coop.numba_mlir as numba_coop``
outside the kernel and use the qualified namespace:

.. code-block:: python

   block = numba_coop.this_block()
   thread = cuda.threadIdx.x
   offset = cuda.blockIdx.x * 128
   neighbor = numba_coop.shuffle(
       block, source[offset + thread], mode="rotate", distance=1
   )
   destination[offset + thread] = neighbor

This second fragment also assumes a complete 128-thread block and 128 valid
elements per block. See :func:`cuda.coop.shuffle` and the
:doc:`../programming_guide` for the common/qualified API distinction and
collective participation rules.

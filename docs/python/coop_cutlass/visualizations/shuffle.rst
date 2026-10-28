.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-shuffle:

Shuffle
=======

:func:`cuda.coop.cutlass.shuffle` shifts values within a complete block and returns a
new per-thread payload. The input remains unchanged. ``up`` and ``down`` move
one item along the flattened blocked tile, including across thread boundaries.
Scalar ``offset`` and ``rotate`` modes operate on one value per thread.
See :func:`cuda.coop.cutlass.shuffle` for the supported distances.

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

The scalar modes use one scalar per thread. ``offset`` reads
thread ``t + distance``, with an undefined result when that thread lies
outside the block. ``rotate`` reads ``(t + distance) % block_size`` and
requires ``1 <= distance < block_size``. Both accept runtime distances,
which may differ between threads; the explorer uses a uniform distance.

Select a value to follow its source and destination. The shared row shows
the values communicated between threads, not a bank layout or a count of
hardware instructions.

Using Shuffle in a kernel
-------------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads and provide at
least ``128 * items_per_thread`` source and destination elements per block.

.. code-block:: python

   block = coop.this_block()
   offset = block_index * 128 * items_per_thread
   items = coop.load(
       block, source, items_per_thread=items_per_thread, offset=offset
   )
   shifted = coop.shuffle(block, items, mode="up")
   for item in range(items_per_thread):
       position = thread_rank * items_per_thread + item
       if position > 0:
           destination[offset + position] = shifted[item]

Every thread calls Shuffle. The conditional guards only the write afterward,
so the undefined first result is never consumed. The first element of each
block's destination tile is left unchanged; initialize those elements
separately if they need a value.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_collective_examples.py
   :language: python
   :start-after: # qualified-rotate-example-begin
   :end-before: # qualified-rotate-example-end
   :dedent: 4

See :func:`cuda.coop.cutlass.shuffle` and
:ref:`the Programming Guide <coop-cutlass-shuffle>` for scalar and array modes.

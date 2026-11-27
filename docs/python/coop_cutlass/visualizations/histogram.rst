.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-histogram:

Histogram
=========

A histogram counts how many of a block's samples fall in each bin. Each
sample is itself a bin index: a sample with value ``b`` adds one to bin
``b``.

Choose **One tile, striped result** to see which threads receive the counters
from :func:`~cuda.coop.cutlass.histogram`. Choose **Shared bins across two
tiles** to see :func:`~cuda.coop.cutlass.histogram_composite` update the
same bins twice. Both modes use one block of eight teaching threads.

.. coop-visualization:: histogram

   .. only:: html

      .. figure:: histogram.svg
         :alt: Eight threads hold two samples each. The counts of bins 0 to
               12 are 2, 1, 1, 2, 1, 1, 2, 0, 1, 2, 1, 1, 1. Each thread
               receives two counters in striped order, and the counters
               that stand for bins 13 to 15 are zero.
         :width: 100%

         Thirteen bins, two samples and two counters per thread.

   The thirteen counts add up to sixteen, the number of samples. Thread
   ``t`` receives the counts of bins ``t`` and ``t + 8``. A dash marks a
   bin index past the last bin, and the counter there is zero.

   .. code-block:: text

      Thread           T0   T1   T2   T3   T4   T5   T6   T7
      Input samples   0,5 10,3  8,0 6,11  3,9  1,6 12,4  9,2
      Returned bins   0,8  1,9 2,10 3,11 4,12  5,-  6,-  7,-
      Returned counts 2,1  1,2  1,1  2,1  1,1  1,0  2,0  0,0

      Counts in bin order: 2 1 1 2 1 1 2 0 1 2 1 1 1

Reading the stages
------------------

For a single tile, ``histogram`` first sets its counters to zero. With ``atomic``, every thread
then adds one to the counter that each of its samples names. The additions
are atomic, so every sample is counted when several threads add to one bin.
The explorer shows one round for each item, with one sample from every
thread. On a GPU the additions can happen in any order.

With ``sort``, equal samples are sorted next to each other, and the length
of each run of equal samples is that bin's count. The call copies the
samples before it hands them to CUB, with either algorithm. The explorer
draws the copy for ``sort``, where CUB reorders it, and the input row stays
as it was.

The last stage shows the counters grouped by thread, the way the call
returns them: thread ``t`` holds the counts of bins ``t``, ``t + 8`` and so
on. :ref:`Histogram <coop-cutlass-histogram>` in the API reference has the
rules for each argument and for the result.

Keeping counts across tiles
---------------------------

:func:`~cuda.coop.cutlass.histogram_init` clears the shared bins before
the loop. Each :func:`~cuda.coop.cutlass.histogram_composite` adds its tile
to the counts already there. The table follows one block of eight threads,
with five bins and one sample per thread.

.. list-table:: Shared counters after each call
   :header-rows: 1

   * - Call
     - Tile samples
     - Shared bins 0 through 4
   * - Initialize
     -
     - ``0 0 0 0 0``
   * - Composite tile 1
     - ``0 0 0 1 1 1 2 2``
     - ``3 3 2 0 0``
   * - Composite tile 2
     - ``2 3 3 3 4 4 4 0``
     - ``4 3 3 3 3``

Bin 1 stays at three because the second tile has no ones. Bin 0 rises
from three to four. The :ref:`multi-tile example
<coop-faq-histogram-accumulation>` writes the shared bins to global
memory after the last tile.

Using Histogram in a kernel
---------------------------

This tested kernel counts one tile on a block of 64 threads into 65 bins,
with two ``int64`` counters per thread. The striped store writes 128
counters: the 65 counts in bin order, then 63 zeros.

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_histogram.py>`__

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_histogram.py
   :language: python
   :start-after: docs: start cutlass-histogram
   :end-before: docs: end cutlass-histogram
   :dedent: 4

:ref:`Histogram <coop-cutlass-histogram>` in the API reference covers both
algorithms, counting several tiles and counting a partial tile.

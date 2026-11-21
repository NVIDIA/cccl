.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Histogram
=========

Histogram counts each input byte in a bin with the same integer index. A call
starts its own zeroed histogram and returns counts for that Tile only.

Interface and examples
----------------------

For a Tile of unsigned bytes, both calls count all 256 possible byte values.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto counts = coop::histogram<256>(samples);
   auto same_counts = coop::histogram_256(samples);

The input has element type ``unsigned char``. Choose 32, 64, 128, or 256
bins at compile time. Every sample must be less than the bin count. The
compiler must find a worker count that divides both the input extent
and the bin count. An optional ``histogram_policy<>`` uses the defaults.

Behavior
--------

The result has type ``int`` and one element per bin. Element ``b`` counts
the inputs equal to ``b``. A bin with no samples contains zero. All the
counters together sum to the number of input samples. Each call starts
its counters from zero.

.. figure:: /tile/visualizations/histogram.svg
   :alt: A 256-sample byte Tile produces 256 counts with 64 occupied bins.

   The 256 input samples occupy 64 bins. All 256 output bins are present,
   including zero counts for values that do not occur.

:ref:`Histogram declarations and overloads <tile-cpp-histogram-declarations>`.

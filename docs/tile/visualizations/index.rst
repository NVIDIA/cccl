.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Tile data-flow diagrams
=======================

Read each row from left to right as positions in a logical Tile. Exchange,
Histogram, and Run Length Decode show the extents from the executable
examples, with ellipses for omitted positions. The other figures use
small inputs so you can follow every value. Each caption links to the
language references. Links to CUB describe the underlying primitives.

Prefixes and reduction
----------------------

.. figure:: scan.svg
   :alt: Exclusive and inclusive prefixes with seeds and different operations.

   Each prefix has a different endpoint. An initial or prefix value supplies
   the value before the first input. See :doc:`Python </tile/python/primitives/scan>` and
   :doc:`C++ </tile/cpp/primitives/scan>`.
   The block scan uses :cpp:class:`cub::BlockScan`.

.. figure:: reduce.svg
   :alt: Sum, minimum, maximum, and product broadcast over the entire output.

   Python ``broadcast=True`` and C++ reductions without an axis repeat the
   aggregate at every input position. Python reductions return only the
   aggregates by default; C++ axis overloads preserve rank with an extent of
   one on the reduced axis. See :doc:`Python </tile/python/primitives/reduce>` and
   :doc:`C++ </tile/cpp/primitives/reduce>`.
   Depending on the reduction shape, the provider can use
   :cpp:class:`cub::WarpReduceBatched`, :cpp:class:`cub::WarpReduce`,
   or :cpp:class:`cub::BlockReduce`.

Movement and neighbors
----------------------

.. figure:: exchange-shuffle.svg
   :alt: Exchange permutations, fixed shifts, offset reads, and wraparound rotation.

   Exchange uses an explicit block width to define the permutation. Offset
   preserves out-of-range destinations; rotation wraps them.
   See :doc:`Python </tile/python/primitives/exchange_shuffle>` and
   :doc:`C++ </tile/cpp/primitives/exchange_shuffle>`.
   The provider implements these movements with :cpp:class:`cub::BlockExchange`
   and :cpp:class:`cub::BlockShuffle`.

.. figure:: adjacent-discontinuity.svg
   :alt: Neighbor subtraction, head and tail masks, and one-based run IDs.

   Discontinuity compares adjacent values and marks runs. An inclusive sum
   of the head mask numbers those runs. See
   :doc:`Python </tile/python/primitives/adjacent_discontinuity>` and
   :doc:`C++ </tile/cpp/primitives/adjacent_discontinuity>`.
   The corresponding CUB primitives are :cpp:class:`cub::BlockAdjacentDifference`
   and :cpp:class:`cub::BlockDiscontinuity`.

Counts and decoded windows
--------------------------

.. figure:: histogram.svg
   :alt: A 256-sample input produces 256 bin counts, including empty bins.

   Histogram output extent is the bin count. Each byte sample selects one
   bin to increment. See :doc:`Python </tile/python/primitives/histogram>` and
   :doc:`C++ </tile/cpp/primitives/histogram>`.
   The provider counts the samples with :cpp:class:`cub::BlockHistogram`.

.. figure:: run-length-decode.svg
   :alt: A decode window starts within a run and pads beyond the expanded sequence.

   Decode output extent is the requested capacity. Relative offsets count
   within each run and mark padding with the unsigned maximum. See
   :doc:`Python </tile/python/primitives/run_length_decode>` and
   :doc:`C++ </tile/cpp/primitives/run_length_decode>`.
   The underlying CUB primitive is :cpp:class:`cub::BlockRunLengthDecode`.

Ordering and selection
----------------------

.. figure:: merge-sort.svg
   :alt: Full and partial comparison sorting with associated payload values.

   Padding keys sort after the meaningful prefix when the caller selects a
   suitable sentinel. See :doc:`Python </tile/python/primitives/merge_sort>` and
   :doc:`C++ </tile/cpp/primitives/merge_sort>`.
   The provider uses :cpp:class:`cub::BlockMergeSort` or :cpp:class:`cub::WarpMergeSort`
   for the selected implementation.

.. figure:: radix-sort-rank.svg
   :alt: Stable radix sorting and per-input destination ranks for a bit interval.

   Rank is indexed by the original input position. Sorting returns keys in
   the selected digit order. See :doc:`Python </tile/python/primitives/radix_sort_rank>` and
   :doc:`C++ </tile/cpp/primitives/radix_sort_rank>`.
   See :cpp:class:`cub::BlockRadixSort` for sorting and :cpp:class:`cub::BlockRadixRank`
   for assigning each position a rank.

.. figure:: topk.svg
   :alt: Maximum and minimum TopK selection with a defined prefix and an unspecified tail.

   Only the first ``min(k, valid_items)`` outputs are defined. Their order and
   tied payload identity are unspecified. See :doc:`Python </tile/python/primitives/topk>` and
   :doc:`C++ </tile/cpp/primitives/topk>`.

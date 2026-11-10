.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Radix sort and rank
===================

Radix sort orders keys by a selected interval of key bits. Radix rank computes
where each input item belongs for a single radix digit, without returning a
reordered key Tile. Pair sorting carries values alongside their keys.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

Sort accepts numeric keys and optional numeric payloads of the same positive
extent ``N <= 65535``; output shapes and types match their inputs. Its runtime
bit interval must satisfy ``0 <= begin_bit <= end_bit <= key_bits``. Sort defaults
to all key bits and ascending order. Sort processes ``RadixBits`` in ``[1, 8]``
per pass; that capacity does not limit the total interval sorted.

Rank accepts only signed or unsigned integer keys, requires ``N <= 65535``,
and returns an ``int`` Tile of extent ``N``. It computes a stable destination rank for each original input
position. Require ``0 <= begin_bit < end_bit <= key_bits`` and
``end_bit - begin_bit <= RadixBits``. Without explicit bounds, rank begins at
zero and ends at the smaller of ``RadixBits`` and the key width.
Physical participation is selected for the logical extent.

``radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>``
defaults to ``4, true, scan_algorithm::warp_scans,
radix_smem_config::four_byte``. Ascending and descending sort overloads accept
``cuda::std::less`` and ``cuda::std::greater`` functors. Bit bounds follow the
comparison when present, and the policy is the final argument. Sort and rank
bit bounds accept integer scalars or shape-one ``int`` Tiles; other Tile
shapes and element types do not match those control overloads.

``radix_rank_policy<RadixBits, Descending, MemoizeOuterScan,
InnerScanAlgorithm, SmemConfig>`` additionally encodes the rank direction;
``Descending`` defaults to ``false``. The other defaults match radix sort,
but rank allows a radix capacity up to 16 bits.

See :ref:`Radix scratch capacity <tile-radix-capacity>` for resource limits
on these policies.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto ordered = coop::radix_sort_keys(keys, 0, 8);
   auto [ordered_keys, ordered_values] = coop::radix_sort_pairs(keys, values);
   auto ranks = coop::radix_rank_keys(keys, 0, 4, coop::radix_rank_policy<4>{});


Behavior
--------


Signed integers and floating-point keys use CUB's radix ordering transform
before the bit interval is selected. A partial interval therefore refers to
that transformed representation, not raw floating-point bits. Use unsigned
integers for direct bit-field sorting. Full-width floating-point sorting
handles signed numeric order; NaNs follow their transformed representations,
not a user-defined total ordering. Equal selected digits preserve input order
under radix sorting.

The rank width is a single-digit capacity. A full sort over a larger bit
interval requires ``radix_sort_keys`` or ``radix_sort_pairs``.
Rank does not sort or mutate its input. Signed integer keys have their sign
bit flipped before digit extraction, matching CUB's radix sort ordering.
An interval containing the sign bit
therefore places negative keys before nonnegative keys in ascending order.
Unsigned keys use their original bits.

.. figure:: /tile/visualizations/radix-sort-rank.svg
   :alt: Stable radix sort and per-input digit ranks with corresponding payloads.

   A rank is associated with an original input position. Sorted output is
   in selected-digit order and preserves equal-digit input order.

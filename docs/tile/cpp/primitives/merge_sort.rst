.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Merge sort
==========

Merge sort orders all keys in one Tile by numeric comparison. The pair form
moves a payload with each key. A valid-item count can limit the meaningful
input prefix while retaining the Tile's compile-time capacity.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

Keys and optional payloads are numeric Tiles with the same positive extent
``N``; key and payload types may differ. Outputs retain their respective input
types and extent. The default comparison is ascending. For partial sorting,
provide both ``valid_items`` in ``[0, N]`` and ``oob_default``. Every input key
after the valid prefix is replaced by that padding key before sorting. Select
a strict high sentinel for ascending order or a strict low sentinel for
descending order when the first ``valid_items`` outputs must be the valid data.
Physical participation is selected for the logical extent.

``merge_sort_keys`` and ``merge_sort_pairs`` accept optional
``cuda::std::less`` or ``cuda::std::greater`` comparison functors and a
``merge_sort_policy<>`` with no options. Typed functors must match the key
type. Custom comparators are unsupported. Overloads with a partial prefix
take the valid count and padding key after the comparison, when one is supplied. The valid
count accepts an integer scalar or a shape-one ``int`` Tile. The padding key
accepts a scalar convertible to the key type or a shape-one Tile of exactly
the key type.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto ascending = coop::merge_sort_keys(keys);
   auto [ordered_keys, ordered_values] = coop::merge_sort_pairs(
     keys, values, cuda::std::greater<>{});
   auto partial = coop::merge_sort_keys(keys, 193, 2147483647);


Behavior
--------


The pair form zeroes padded payloads before sorting. Read the first
``valid_items`` results only after choosing a sentinel that sorts after the
valid keys. If valid keys equal the sentinel, padded and valid payloads may
interleave at that key: choose a strict sentinel when payload identity matters.
With a strict sentinel and a well-defined comparison order, the trailing
``N - valid_items`` keys equal ``oob_default`` and their payloads are zero.
The output capacity remains ``N`` even when ``valid_items`` is zero.

Merge sort does not guarantee stable ordering of equal keys. Numeric comparison
is used directly; do not assume a total ordering for NaNs. Use inputs with a
well-defined comparison order when comparing against a sorted reference.
The provider selects a power-of-two worker count dividing ``N``. The special
warp implementation is limited to exactly 32 input items and 32 workers;
those properties are derived from the logical extent, not a per-call policy.
A compiler/frontend supporting that worker count is still required. The
block examples use 256 items with 128 workers, including
full and partial keys and pairs. These implementation constraints do not
expose an items-per-thread option.

.. figure:: /tile/visualizations/merge-sort.svg
   :alt: Sorted keys retain payload association, including a partial valid prefix.

   Sorted keys retain payload association, including a partial valid prefix.

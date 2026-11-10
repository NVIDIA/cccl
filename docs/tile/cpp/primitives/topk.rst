.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

TopK selection
==============

TopK selects the largest or smallest ``k`` keys from a Tile, optionally carrying
payloads. The result has the original capacity, and only its first
``min(k, valid_items)`` positions are defined. Selected keys are not guaranteed
to be sorted.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

Keys and optional payloads are numeric Tiles with matching positive extent
``N``; their element types may differ. The results retain capacity ``N`` and
the respective input types. Require ``0 <= k <= N`` and
``0 <= valid_items <= N``; omitting ``valid_items`` selects the whole input. Keys and payloads beyond that input
prefix are excluded. Bit bounds default to the whole key and must satisfy
``0 <= begin_bit < end_bit <= key_bits``. They select the radix ordering
representation described in :doc:`/tile/cpp/primitives/radix_sort_rank`. Physical participation
is selected for the logical extent.

The four functions have the same names in ``cuda::coop::tile``. They take
``k`` after the data arguments and optionally ``valid_items``, bit bounds, and a
final empty ``topk_policy<>``. The numerical controls are
integer scalars; shape-one Tile controls are unsupported. Pair results use a
structured binding, with keys first and values second.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto smallest = coop::topk_min_keys(keys, 16);
   auto [best_keys, best_values] = coop::topk_max_pairs(
     keys, values, 16, 193, 0, 32);
   // Only positions [0, 16) of each result are defined.


Behavior
--------


Only the first ``min(k, valid_items)`` output slots may be consumed. The remaining
slots are unspecified; they are not zero padding. Pair association is
preserved within the selected prefix. Ties at the cutoff may choose different
input items, so do not assert a particular payload for equal keys. For
``k >= valid_items``, all valid items are selected, but their order remains
unspecified. When ``k == 0`` or ``valid_items == 0``, no output positions are
defined. Counts outside ``[0, N]`` are invalid.

Selecting a smaller bit interval compares only that transformed digit. It can
change which values are selected even when their full numeric values differ.
Use full-width bounds for ordinary numeric TopK.

.. figure:: /tile/visualizations/topk.svg
   :alt: Maximum and minimum selection define only a leading prefix of the output.

   Maximum and minimum selection define only a leading prefix of the output.

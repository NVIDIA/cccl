.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

TopK selection
==============

.. py:currentmodule:: cuda.coop.tile

TopK selects the largest or smallest ``k`` keys from a Tile, optionally carrying
payloads. The result has the original capacity, and only its first
``min(k, valid_items)`` positions are defined. Selected keys are not guaranteed
to be sorted.

Interface and examples
----------------------

.. py:function:: topk_max_keys(keys, /, *, k, valid_items=None, begin_bit=None, end_bit=None, methods=None)
   :no-contents-entry:

   Select the ``k`` largest keys from the valid input prefix.

.. py:function:: topk_min_keys(keys, /, *, k, valid_items=None, begin_bit=None, end_bit=None, methods=None)
   :no-contents-entry:

   Select the ``k`` smallest keys from the valid input prefix.

.. py:function:: topk_max_pairs(keys, values, /, *, k, valid_items=None, begin_bit=None, end_bit=None, methods=None)
   :no-contents-entry:

   Return ``(selected_keys, selected_values)`` for the ``k`` largest keys.

.. py:function:: topk_min_pairs(keys, values, /, *, k, valid_items=None, begin_bit=None, end_bit=None, methods=None)
   :no-contents-entry:

   Return ``(selected_keys, selected_values)`` for the ``k`` smallest keys.

.. rubric:: Parameters and results

``keys``, ``values``
   Numeric Tiles of shape ``(N,)``. Pair shapes must match, but their types
   may differ. Results preserve key and payload types and shape ``(N,)``.
``k``
   Number of items requested, satisfying ``0 <= k <= N``.
   It may be a compile-time integer, runtime integer scalar, or rank-zero
   integer Tile. A shape-one Tile is rejected. The return type does not
   change when ``k`` changes at runtime.
``valid_items``
   Length of the meaningful input prefix, defaulting to ``N``. Require
   ``0 <= valid_items <= N``. Keys and payloads after this prefix are excluded
   from selection. The scalar restrictions are the same as for ``k``.
``begin_bit``, ``end_bit``
   Half-open interval in the radix ordering representation. Defaults are
   zero and the key width. Require
   ``0 <= begin_bit < end_bit <= key_bits``. Bounds may be runtime integer
   scalars or rank-zero Tiles, with the same shape restriction as ``k``.
   Signed and floating-point transformations match :doc:`/tile/python/primitives/radix_sort_rank`.
``methods``
   Accepts only ``begin_bit`` and ``end_bit`` as compile-time options.
   The function name selects maximum or minimum; ``k`` and ``valid_items``
   remain direct arguments.

.. rubric:: Example: select keys and source positions

The kernel stores full-capacity results. The host consumes only the first
16 entries and compares the selected set without relying on output order.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def topk_kernel(keys, ids, selected_keys, selected_ids):
       k = ct.load(keys, (ct.bid(0),), shape=(256,))
       v = ct.load(ids, (ct.bid(0),), shape=(256,))
       best_k, best_v = coop.topk_max_pairs(k, v, k=16, valid_items=193)
       ct.store(selected_keys, (ct.bid(0),), best_k)
       ct.store(selected_ids, (ct.bid(0),), best_v)

   ids = torch.arange(256, device="cuda", dtype=torch.int32)
   keys = (ids * 73) % 256
   selected_keys = torch.empty_like(keys)
   selected_ids = torch.empty_like(ids)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), topk_kernel,
             (keys, ids, selected_keys, selected_ids))
   actual = selected_keys[:16]
   expected = keys[:193].topk(16).values
   assert torch.equal(actual.sort().values, expected.sort().values)
   assert torch.all(selected_ids[:16] < 193)
   assert torch.equal(actual, keys[selected_ids[:16].to(torch.int64)])

To sort selected keys afterward, preserve the valid-prefix contract: use a
sorting primitive with an appropriate padding key or a consumer that operates
only on the selected prefix. Sorting all ``N`` TopK output slots would consume
undefined values.

.. rubric:: Combining candidates from several Tiles

To find eight results across four column Tiles, each Tile can first retain its
own eight best key/value pairs. A second TopK then selects eight of those
32 candidates. The local winners contain enough candidates for a valid global
TopK, so this avoids storing the full set of intermediate scores. Tied payload
identity remains unspecified.

The :doc:`MMA composition example </tile/python/programming>` uses this two-stage scheme.
Its merge loads each 32-element candidate row into a 128-element Tile with
zero padding and four worker warps. It passes
``valid_items=partial_scores.shape[1]`` to TopK, using the runtime array width
to exclude the 96 padded positions. The output still has capacity 128;
the example extracts only the first eight results before storing them.


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

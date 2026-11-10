.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Merge sort
==========

.. py:currentmodule:: cuda.coop.tile

Merge sort orders all keys in one Tile by numeric comparison. The pair form
moves a payload with each key. A valid-item count can limit the meaningful
input prefix while retaining the Tile's compile-time capacity.

Interface and examples
----------------------

.. py:function:: merge_sort_keys(keys, /, *, descending=None, valid_items=None, oob_default=None, methods=None)
   :no-contents-entry:

   Return keys in ascending order, or descending order when requested.

.. py:function:: merge_sort_pairs(keys, values, /, *, descending=None, valid_items=None, oob_default=None, methods=None)
   :no-contents-entry:

   Return ``(sorted_keys, sorted_values)`` with key/value association preserved.

.. rubric:: Parameters and results

``keys``
   A numeric Tile of shape ``(N,)``. The returned key Tile has the same shape
   and dtype. Sorting does not change the original Tile.
``values``
   For the pair form, a numeric payload Tile of the same shape as ``keys``.
   Its element type may differ. The returned payload preserves its dtype.
``descending``
   A compile-time boolean. ``None`` selects the default ascending order;
   ``True`` reverses the comparison.
``valid_items``, ``oob_default``
   Supply both or neither. ``valid_items`` is an integer in ``[0, N]`` and
   may be a runtime scalar or single-element Tile. Only that many leading
   keys are meaningful. The provider replaces every other key with
   ``oob_default`` before sorting. The padding value converts to the key
   type and may also be runtime. Use a key greater than all valid keys for
   ascending order or less than all valid keys for descending order.
``methods``
   Accepts ``descending``, ``valid_items``, and ``oob_default``.

.. rubric:: Example: sort keys and original positions

The 256 unique keys below are a permutation of ``[0, 255]``. The payload lets
the host check that each output key still refers to its original input slot.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def sort_kernel(keys, ids, sorted_keys, sorted_ids):
       k = ct.load(keys, (ct.bid(0),), shape=(256,))
       v = ct.load(ids, (ct.bid(0),), shape=(256,))
       ordered_k, ordered_v = coop.merge_sort_pairs(k, v)
       ct.store(sorted_keys, (ct.bid(0),), ordered_k)
       ct.store(sorted_ids, (ct.bid(0),), ordered_v)

   ids = torch.arange(256, device="cuda", dtype=torch.int32)
   keys = (ids * 73) % 256
   sorted_keys = torch.empty_like(keys)
   sorted_ids = torch.empty_like(ids)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), sort_kernel,
             (keys, ids, sorted_keys, sorted_ids))
   assert torch.equal(sorted_keys, ids)
   assert torch.equal(sorted_keys, keys[sorted_ids.to(torch.int64)])

For a partial integer Tile, use for example
``coop.merge_sort_keys(k, valid_items=193, oob_default=2147483647)`` when every
valid key is strictly below that sentinel, then consume only the first 193
results.


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

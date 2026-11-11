.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Radix sort and rank
===================

.. py:currentmodule:: cuda.coop.tile

Radix sort orders keys by a selected interval of key bits. Radix rank computes
where each input item belongs for a single radix digit, without returning a
reordered key Tile. Pair sorting carries values alongside their keys.

Interface and examples
----------------------

.. py:function:: radix_sort_keys(keys, /, *, begin_bit=None, end_bit=None, descending=None, radix_bits=None, memoize_outer_scan=None, inner_scan_algorithm=None, smem_config=None, methods=None)
   :no-contents-entry:

   Return sorted keys, retaining the input capacity and dtype.

.. py:function:: radix_sort_pairs(keys, values, /, *, begin_bit=None, end_bit=None, descending=None, radix_bits=None, memoize_outer_scan=None, inner_scan_algorithm=None, smem_config=None, methods=None)
   :no-contents-entry:

   Return ``(sorted_keys, sorted_values)`` with the input pair association.

.. py:function:: radix_rank_keys(keys, /, *, begin_bit=0, end_bit=None, radix_bits=None, descending=False, memoize_outer_scan=None, inner_scan_algorithm=None, smem_config=None, methods=None)
   :no-contents-entry:

   Return an ``int32`` destination rank for each input key.

.. rubric:: Sort parameters

``keys``, ``values``
   Numeric Tiles of shape ``(N,)`` with ``N <= 65535``. Pair shapes must match;
   key and payload types may differ. Output Tiles preserve the corresponding
   input types and shape.
``begin_bit``, ``end_bit``
   The half-open bit interval ``[begin_bit, end_bit)``. Defaults are zero and
   the key width. Bounds may be runtime scalars or single-element Tiles.
   Require ``0 <= begin_bit <= end_bit <= key_bits``. An empty interval leaves
   all keys equivalent under the radix comparison.
``descending``
   A compile-time boolean; the default is ascending order.
``radix_bits``
   The compile-time number of bits processed per pass, in ``[1, 8]``.
   The default is 4. It does not limit the total interval sorted.
``memoize_outer_scan``
   Compile-time boolean, default ``True``. Selects whether the CUB radix
   implementation memoizes its outer scan.
``inner_scan_algorithm``
   ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"`` (default).
``smem_config``
   ``"four_byte"`` (default) or ``"eight_byte"``. ``"default"`` selects
   four-byte banks; integer 4/8 and string ``"4"``/``"8"`` are accepted
   aliases. This is an implementation setting, not an element-type conversion.
``methods``
   Accepts the seven option names above: bit bounds, descending order,
   radix bits, memoization, inner scan algorithm, and shared-memory configuration.

.. rubric:: Rank parameters and results

``radix_rank_keys`` accepts only signed or unsigned integer keys of width
8, 16, 32, or 64, with ``N <= 65535``. The output has shape ``(N,)`` and dtype
``int32``. Output ``rank[i]`` is the destination of input item ``i`` in the stable ordering of
the selected digit; equal digits retain their input order. For
``[3, 1, 3, 0]`` with bits ``[0, 2)``, ranks are ``[2, 1, 3, 0]``.

``begin_bit`` defaults to 0. ``radix_bits`` defaults to 4 and is a compile-time
capacity in ``[1, 16]``. ``end_bit`` defaults to
``min(key_bits, begin_bit + radix_bits)``. Bounds may be runtime scalar or
single-element Tiles, but must satisfy
``0 <= begin_bit < end_bit <= key_bits`` and
``end_bit - begin_bit <= radix_bits``. ``descending`` is a compile-time boolean
with a default of ``False``. ``memoize_outer_scan``, ``inner_scan_algorithm``,
and ``smem_config`` have the same defaults and accepted values as radix sort.
``methods`` must be omitted or empty.

See :ref:`Radix scratch capacity <tile-radix-capacity>` for resource limits
on these policies.

.. rubric:: Example: sort the low byte and rank its low nibble

This complete example uses 128 nonnegative ``int32`` keys. The selected low
bits correspond directly to their integer digits.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def radix_kernel(source, ordered, ranks):
       keys = ct.load(source, (ct.bid(0),), shape=(128,))
       sorted_keys = coop.radix_sort_keys(keys, begin_bit=0, end_bit=8)
       digit_ranks = coop.radix_rank_keys(keys, begin_bit=0, end_bit=4, radix_bits=4)
       ct.store(ordered, (ct.bid(0),), sorted_keys)
       ct.store(ranks, (ct.bid(0),), digit_ranks)

   source = (torch.arange(128, device="cuda", dtype=torch.int32) * 73) % 256
   ordered = torch.empty_like(source)
   ranks = torch.empty_like(source)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), radix_kernel,
             (source, ordered, ranks))
   assert torch.equal(ordered, source.sort().values)
   permutation = torch.argsort(source & 15, stable=True)
   expected_ranks = torch.empty_like(ranks)
   expected_ranks[permutation] = torch.arange(128, device="cuda", dtype=torch.int32)
   assert torch.equal(ranks, expected_ranks)


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

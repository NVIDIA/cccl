.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-radix:

Radix Sort and Rank
===================

:func:`cuda.coop.radix_sort_keys` and :func:`cuda.coop.radix_sort_pairs`
order a block's keys by selected bits. Sort processes digits from the
least significant selected bits upward, moving keys after each stable pass.
The pairs operation moves associated values with their keys.

:func:`cuda.coop.radix_rank` computes destinations for **one digit**. It
returns an ``int32`` rank in each key's original slot. The input keys stay
where they were; ranking does not perform a scatter. All three operations
preserve their input payloads.

The explorer uses eight teaching threads with unsigned ``uint32`` keys.
Only their low eight bits are shown. The examples below the explorer use
complete, executable blocks of 64 threads.

.. coop-visualization:: radix

   .. only:: html

      .. figure:: radix-sort.svg
         :alt: Keys 35, 18, 33, 2, 35, 17, 1, 18 sort by their low four bits and then their next four bits. Associated original positions move with each key, preserving the order of equal keys.
         :width: 100%

         Two stable digit passes sort eight unsigned keys. Values identify
         the original positions of the keys.

   For ``begin_bit=0, end_bit=8``, the low digit orders bits zero through
   three. The next pass orders bits four through seven, keeping the order
   established by the low digit within each equal high digit.

   .. code-block:: text

      Original keys       35  18  33   2  35  17   1  18
      Original values      0   1   2   3   4   5   6   7
      After bits [0, 4)    33  17   1  18   2  18  35  35
      Associated values    2   5   6   1   3   7   0   4
      After bits [4, 8)     1   2  17  18  18  33  35  35
      Associated values    6   3   5   1   7   2   0   4

   .. only:: html

      .. figure:: radix-rank.svg
         :alt: Ranking the low four bits of the same keys returns ranks 6, 3, 0, 4, 7, 1, 2, 5 in the original input slots. The keys remain unchanged.
         :width: 100%

         One digit's ranks describe destinations without moving the keys.

   For ``radix_rank(..., begin_bit=0, end_bit=4)``, digits one, two, and
   three occur three, three, and two times. Their ascending bin starts
   are zero, three, and six. Add the number of earlier keys with the
   same digit to obtain each destination.

   .. code-block:: text

      Input keys          35  18  33   2  35  17   1  18
      Low-four-bit digit   3   2   1   2   3   1   1   2
      Returned ranks       6   3   0   4   7   1   2   5
      Input after ranking 35  18  33   2  35  17   1  18

Reading the stages
------------------

Input order is :ref:`blocked <coop-glossary-layouts>`: all of thread zero's
items, then all of thread one's items, and so on. For ``K`` items per
thread, item ``i`` of thread ``t`` has input position ``t * K + i``. This
order also determines which equal-digit key comes first.

For ascending Rank, each result is the number of keys with smaller digits
plus the number of earlier input keys with the same digit. For descending
Rank, count greater digits instead. The rank is a unique destination in the
block tile, but it is returned at the key's original blocked position.
The counts and bin starts in the picture explain the calculation; the
common function returns only the per-key ranks.

Sort uses those destinations to move a working copy of the keys, and any
associated values, before examining the next digit. Equal digits retain
their order at each pass. That stability lets the more significant pass
retain the ordering already established by less significant bits.
``descending=True`` reverses digit ordering while preserving the input
order of ties; it does not reverse a finished ascending array.

Both backends use the shared CUB ``BlockRadixSort`` specialization
with four bits per pass. A final pass may use fewer bits. This is an
implementation choice, not an argument to ``radix_sort_keys`` or
``radix_sort_pairs``. The picture shows ranks and ownership changes, not
the exact CUB instruction sequence or shared-memory layout.

The common sort calls return blocked output. The qualified calls in
``cuda.coop.numba_mlir`` and ``cuda.coop.cutlass`` also accept
``blocked_to_striped=True``. With
``T`` block threads, sorted position ``p`` then belongs to thread
``p % T``, slot ``p // T``. Intermediate digit passes still use blocked
ownership. Keys and values use the same output layout; select a matching
layout when storing them. Rank has no striped-output or pairs option.

Bit intervals and key types
---------------------------

``[begin_bit, end_bit)`` is a half-open interval in CUB's ordered key
representation. Unsigned keys use their bits directly, which is why the
explorer uses unsigned keys. Sorting a subset of bits orders that bit field;
the full numeric keys need not be increasing. Bits outside the field do not
break ties.

For signed integers, the sign bit is inverted before digit extraction.
The qualified sort calls also accept ``float32`` and ``float64`` keys:
CUB inverts all bits for negative floating-point keys and the sign bit for
nonnegative keys before selecting the interval. Returned keys retain their
original representations. Negative and positive zero compare equally and
preserve their input order; NaNs follow CUB's transformed-bit ordering.
The unsigned explorer does not model these floating-point cases.

Common calls accept ``ThreadData`` keys with ``int32``, ``uint32``,
``int64``, or ``uint64`` dtype. Associated values may use the common
numeric dtypes and must have the same extent as the keys. Qualified calls
also accept scalars. Numba-qualified calls accept fixed local arrays;
CUTLASS-qualified calls accept CuTe register tensors and ``TensorSSA``.
Floating-point keys are supported for Sort, while Rank requires integral keys.

Sort's bit bounds may be block-uniform runtime integers and must satisfy
``0 <= begin_bit < end_bit <= key_width``. Omitting ``end_bit`` selects the
key width. Rank requires compile-time bounds selecting one through eight
bits. Its omitted end is ``begin_bit + radix_bits``, with four bits when
``radix_bits`` is also omitted. If both ``radix_bits`` and ``end_bit`` are
supplied, they must describe the same digit width. The explorer limits its
controls to the low eight bits of its small example keys.

All threads in the complete physical block must participate with matching
controls and payload extents. These operations do not accept warp groups
or a ``valid_items`` argument. Sort can use explicit ``temp_storage``;
its size and alignment must cover the specialization, and the caller must
synchronize before reuse when ``auto_sync=False``. Rank allocates and
synchronizes its scratch automatically.

The qualified Rank call can additionally write ``exclusive_digit_prefix``.
That side output describes digit bins and has its own per-thread extent,
separate from the returned per-key ranks. See
:func:`cuda.coop.numba_mlir.radix_rank` and
:func:`cuda.coop.cutlass.radix_rank` for the supported output containers,
layout, and undefined tail slots. The explorer's bin rows are mathematical explanations, not an
invocation of that optional output.

Sorting key/value pairs in a kernel
-----------------------------------

This Numba example sorts signed keys and carries their original positions
as values. The stable host reference checks both ordering and association,
including ties.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_radix_examples.py
   :language: python
   :start-after: # radix-sort-example-begin
   :end-before: # radix-sort-example-end
   :dedent: 4

The CuTe example below covers sorting and ranking in the same kernel. It
uses 64 threads with two items each, and ``module`` selects the common or
CUTLASS-qualified API. Its qualified path also checks striped output and
bin prefixes. :download:`Download the complete CuTe example
<../../../../python/cuda_coop/examples/cutlass/radix.py>` for setup and host
checks.

.. literalinclude:: ../../../../python/cuda_coop/examples/cutlass/radix.py
   :language: python
   :start-after: # docs: start cutlass-radix
   :end-before: # docs: end cutlass-radix
   :dedent: 4

Returning one digit's ranks
---------------------------

This Numba example selects the low four bits of unsigned keys. The CuTe
example above also checks ranks against an independent host reference. The host
reference inverts the stable digit-sort permutation because each rank must
be returned at its original input position.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_radix_examples.py
   :language: python
   :start-after: # radix-rank-example-begin
   :end-before: # radix-rank-example-end
   :dedent: 4

See the :ref:`Numba <coop-radix>` and :ref:`CUTLASS <coop-cutlass-radix>`
guides for runtime setup, and the :doc:`../../coop_api` reference for
common and qualified contracts.

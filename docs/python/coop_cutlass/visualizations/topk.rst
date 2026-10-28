.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-topk:

TopK
====

:func:`cuda.coop.cutlass.topk_min_keys` selects the smallest keys in a block;
:func:`cuda.coop.cutlass.topk_max_keys` selects the largest. The corresponding
:func:`~cuda.coop.cutlass.topk_min_pairs` and :func:`~cuda.coop.cutlass.topk_max_pairs`
operations carry a value with each selected key. All four return new
payloads and preserve the inputs.

The explorer uses one complete one-dimensional block of eight teaching
threads. Change ``k`` and ``valid_items`` to see how the defined output
prefix changes. Repeated keys expose the selection boundary: when several
keys tie there, TopK may choose any subset needed to fill the result.

.. coop-visualization:: topk

   .. only:: html

      .. figure:: topk.svg
         :alt: A block selects four largest keys from twelve valid inputs.
               One permitted unordered result is 11, 14, 9, 12, paired with
               original positions 8, 6, 4, 2. The other twelve positions
               are unspecified.
         :width: 100%

         Maximum pairs, two items per thread, ``k=4``, ``valid_items=12``.

   Only the first ``min(k, valid_items)`` blocked output positions are
   defined. This is one permitted output order. ``?`` marks an unspecified
   output that the program must not read or store; parentheses mark inputs
   outside the valid prefix.

   .. code-block:: text

      Thread             T0    T1    T2    T3    T4    T5     T6      T7
      Input keys         7,2  12,5   9,1  14,4  11,3   8,6  (15,0) (10,13)
      Input positions    0,1   2,3   4,5   6,7   8,9  10,11 (12,13)(14,15)
      Selected keys     11,14  9,12  ?,?   ?,?   ?,?   ?,?    ?,?     ?,?
      Selected positions 8,6   4,2   ?,?   ?,?   ?,?   ?,?    ?,?     ?,?

Reading the stages
------------------

The input rows stay visible throughout playback. In blocked order,
position ``thread_rank * items_per_thread + item`` belongs to that
thread's local ``item`` slot. ``valid_items`` counts the input prefix in
this order, across the whole block. Every thread participates even when
none of its items is valid.

The teaching model uses small nonnegative integer keys and two-bit digits.
It counts candidates by the high digit first, then refines the bucket
containing the selection boundary using the low digit. Digits on the
preferred side are guaranteed selections; digits on the other side are
excluded. Inspect a key to distinguish these from the remaining boundary
candidates. The candidate row annotates source positions. It does not
represent input data moving during filtering.

This follows the candidate-filtering idea in CUB's internal block TopK
implementation. CUB determines a boundary with radix histograms, then
partitions selected items through shared memory into a blocked output
prefix. The explorer's digit width, tie choices, and output order are
illustrative. It does not show the exact compiled instructions or imply a
full sort. When all valid items are selected, filtering is unnecessary.

The defined output length is ``min(k, valid_items)``. Setting either count
to zero leaves no defined output items. If ``k`` exceeds ``valid_items``,
all valid inputs are selected. Both counts must be uniform across the
block and lie in ``[0, block_threads * items_per_thread]``. Uniformity is a
caller precondition. The backend validates count ranges, but does not
check agreement among threads.

TopK does not promise sorted output, stable ordering among equal keys, or
a particular choice of tied items at the boundary. For pairs, the chosen
values still belong to their original keys. Use a sorting primitive when
you need an ordered result. See :ref:`the TopK ordering FAQ
<coop-faq-topk-order>`.

Using TopK in a kernel
----------------------

This example selects the smallest keys and largest pairs from a partial
tile. It verifies selected multisets and pair identities without relying on
output order or stable ties, and demonstrates register-payload conversions.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python
   :start-after: # docs: start cutlass-topk
   :end-before: # docs: end cutlass-topk
   :dedent: 4

Common TopK calls accept numeric ``ThreadData`` payloads in blocked order.
All four variants accept ``ThreadData`` and CuTe register payloads. They
preserve inputs and return fresh payloads. See :ref:`the Programming Guide
<coop-cutlass-topk>` and :doc:`../api` for counts, dtypes, and scratch rules.

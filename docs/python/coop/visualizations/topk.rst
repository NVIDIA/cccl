.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-topk:

TopK
====

:func:`cuda.coop.topk_min_keys` selects the smallest keys in a block;
:func:`cuda.coop.topk_max_keys` selects the largest. The corresponding
:func:`~cuda.coop.topk_min_pairs` and :func:`~cuda.coop.topk_max_pairs`
operations carry a value with each selected key. All four return new
payloads and preserve the inputs.

The explorer uses one complete one-dimensional block of eight teaching
threads. Change ``k`` and ``valid_items`` to see how the defined output
prefix changes. Repeated keys expose the selection boundary: when several
keys tie there, TopK may choose any subset needed to fill the result.

.. coop-visualization:: topk

   .. only:: html

      .. figure:: topk.svg
         :alt: A block selects four largest keys from twelve valid inputs. One permitted unordered result is 11, 14, 9, 12, paired with original positions 8, 6, 4, 2. The other twelve output positions are unspecified.
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

This tested example selects eight largest keys from 93 valid inputs in a
128-item tile. Each value is the key's original position. The kernel uses
64 threads and two items per thread; the host verifies membership and
pair association without depending on output order.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_topk_examples.py
   :language: python
   :start-after: # topk-example-begin
   :end-before: # topk-example-end
   :dedent: 4

The example knows that at least eight inputs are valid. A general kernel
must store only ``min(k, valid_items)`` results, using that value as
Store's ``valid_items``. The result payloads retain the input extent, so
their size alone does not tell you which slots are safe to read.

Common TopK calls accept numeric ``ThreadData`` payloads in blocked order.
The qualified :func:`cuda.coop.numba_mlir.topk_max_pairs` API also accepts
fixed local arrays; its :func:`~cuda.coop.numba_mlir.topk_min_pairs`,
:func:`~cuda.coop.numba_mlir.topk_min_keys`, and
:func:`~cuda.coop.numba_mlir.topk_max_keys` variants have the same selection
contract. The current backend supports complete one-dimensional blocks;
Warp and logical-warp TopK are unsupported.

Scratch is allocated automatically, or supplied through ``temp_storage``.
Follow the descriptor's synchronization requirements before reusing it.
See :doc:`../programming_guide` for scratch reuse and the API reference
for supported dtypes and runtime count types. The explorer omits
floating-point special values: positive and negative zero compare as
equal while retaining their original bits, and NaNs have no guaranteed
numeric ordering.

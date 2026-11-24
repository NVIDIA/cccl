.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-topk:

TopK
====

A top-k call selects the smallest or largest keys of a block's :term:`tile`
and returns them in the first positions of the result, in no defined order.
:func:`~cuda.coop.cutlass.topk_min_pairs` and
:func:`~cuda.coop.cutlass.topk_max_pairs` also return the value that came
with each selected key.

The explorer covers all four functions. It draws one block of eight teaching
threads with one, two or four items each. Change ``k`` and ``valid_items``
to see which positions of the result hold a selected key. With
``Repeated keys`` or ``All keys equal``, several equal keys can compete for
the last selected positions, and the explorer shows one of the selections
that top-k may return.

.. coop-visualization:: topk

   .. only:: html

      .. figure:: topk.svg
         :alt: Eight threads hold two keys each. The four largest of the
               first twelve keys, 11, 14, 9 and 12, fill the first four
               positions of the result with their tile positions 8, 6, 4
               and 2. The other twelve positions of the result are
               question marks.
         :width: 100%

         ``topk_max_pairs`` with two items per thread, ``k=4`` and
         ``valid_items=12``.

   The four largest of the first twelve keys are 14, 12, 11 and 9. They
   fill the first four positions of the result, here in the order 11, 14,
   9, 12, which is one of the orders top-k may return. Each key's value is
   the position it came from. ``?`` marks a position that holds no selected
   key, and parentheses mark the inputs at or past ``valid_items``.

   .. code-block:: text

      Thread             T0    T1    T2    T3    T4    T5     T6      T7
      Input keys         7,2  12,5   9,1  14,4  11,3   8,6  (15,0) (10,13)
      Input positions    0,1   2,3   4,5   6,7   8,9  10,11 (12,13)(14,15)
      Selected keys     11,14  9,12  ?,?   ?,?   ?,?   ?,?    ?,?     ?,?
      Selected positions 8,6   4,2   ?,?   ?,?   ?,?   ?,?    ?,?     ?,?

Reading the stages
------------------

The input rows stay in view during playback. In blocked order, item ``i`` of
thread ``t`` is at position ``t * items_per_thread + i`` of the tile.
``valid_items`` counts positions in this order across the whole block, and
the explorer fades the inputs at or past it.

The explorer is a teaching model with small non-negative keys and two-bit
digits. The ``High digit`` stage counts the candidates by their high digit
and finds the bucket that holds the last key to select. Keys in buckets
with a better digit, lower for the smallest keys and higher for the
largest, are selected. Keys in buckets with a worse digit are left out.
The ``Low digit`` stage repeats the count with the low digit for that one
bucket when its keys do not all fit. Select a key in the
``Selection status`` row to see its status: guaranteed selection, boundary
candidate or excluded. That row marks input positions, and items move only
in the last stage, ``Defined output``, into the result rows.

CUB's block top-k follows the same idea with eight-bit digits. It builds
histograms from the most significant digit down to find the bucket that
holds the last key to select, then moves the selected items through shared
memory into the first positions of the result. The tile is never sorted.
When ``k`` is zero or at least ``valid_items``, the explorer goes from
``Inputs`` to ``Defined output`` with no counting stage.

:ref:`Top-k selection <coop-cutlass-topk>` in the API reference has the
rules for both counts, for equal keys and for the positions after the
selected keys. To get the selected keys in order, see
:ref:`this FAQ <coop-faq-topk-order>`.

Using TopK in a kernel
----------------------

This kernel selects the smallest keys and the largest pairs among the first
``valid_items`` positions of a tile. It passes the keys as a register tensor
and their positions as a ``TensorSSA``, and the two calls share one
``TempStorage``. The first three ``store`` calls write ``selected_items``
items of each result, and the last two write the original ``keys`` and
``positions``. In the `complete program on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/topk.py>`__,
``_THREADS`` is 64 and ``_K`` is 31.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python
   :start-after: # docs: start cutlass-topk
   :end-before: # docs: end cutlass-topk
   :dedent: 4

:ref:`Top-k selection <coop-cutlass-topk>` in the API reference covers the
counts, the element types and the scratch.

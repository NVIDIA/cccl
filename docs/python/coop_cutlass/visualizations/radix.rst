.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-radix:

Radix Sort and Rank
===================

A radix sort orders a block's keys by their bits, one digit at a time. A
digit is a run of neighboring bits of a key.
:func:`cuda.coop.cutlass.radix_sort_keys` returns the keys in sorted order,
and :func:`cuda.coop.cutlass.radix_sort_pairs` moves a value with each key.
:func:`cuda.coop.cutlass.radix_rank_keys` looks at one digit and returns
the position each key would take in a sort by that digit.

The explorer shows one block of eight teaching threads with unsigned keys
below 256, and its controls cover the low eight bits. The kernel at the end
of this page runs on a block of 64 threads.

.. coop-visualization:: radix

   .. only:: html

      .. figure:: radix-sort.svg
         :alt: Three rows of eight threads with one key and one value each.
               The input keys are 35, 18, 33, 2, 35, 17, 1, 18 with the
               values 0 to 7. After the pass on bits 0 to 3 the keys are
               33, 17, 1, 18, 2, 18, 35, 35. After the pass on bits 4 to 7
               they are 1, 2, 17, 18, 18, 33, 35, 35.
         :width: 100%

         A pair sort with ``end_bit=8`` takes two passes. Each value is the
         input position of its key.

   With ``begin_bit=0`` and ``end_bit=8``, the first pass orders the keys by
   bits 0 to 3. The second pass orders them by bits 4 to 7. Keys that are
   equal in bits 4 to 7 keep the order the first pass gave them.

   .. code-block:: text

      Original keys       35  18  33   2  35  17   1  18
      Original values      0   1   2   3   4   5   6   7
      After bits [0, 4)    33  17   1  18   2  18  35  35
      Associated values    2   5   6   1   3   7   0   4
      After bits [4, 8)     1   2  17  18  18  33  35  35
      Associated values    6   3   5   1   7   2   0   4

   .. only:: html

      .. figure:: radix-rank.svg
         :alt: The same eight keys with their low four bits under them, 3,
               2, 1, 2, 3, 1, 1, 2, and under those the ranks 6, 3, 0, 4, 7,
               1, 2, 5. The keys are where they were.
         :width: 100%

         The ranks of the same keys by their low four bits.

   With ``radix_rank_keys(block, keys, radix_bits=4)``, the digits 1, 2 and
   3 occur three, three and two times, so their keys begin at ranks 0, 3
   and 6. A key's rank is where its digit begins plus the number of earlier
   keys with the same digit.

   .. code-block:: text

      Input keys          35  18  33   2  35  17   1  18
      Low-four-bit digit   3   2   1   2   3   1   1   2
      Returned ranks       6   3   0   4   7   1   2   5
      Input after ranking 35  18  33   2  35  17   1  18

Reading the stages
------------------

The input is in :ref:`blocked <coop-glossary-layouts>` order: all of thread
0's items, then all of thread 1's, and so on. With ``K`` items per thread,
item ``i`` of thread ``t`` is at position ``t * K + i``. This order decides
which of two keys with equal digits comes first.

In the Radix Rank view, the count row says how often each digit occurs, and
the bin start row says how many keys have a smaller digit. A key's rank is
the bin start of its digit plus the number of earlier keys with the same
digit. In descending order, a bin start counts the keys with a greater
digit. Each rank appears under its own key, and the keys stay where they
were. ``exclusive_digit_prefix`` receives the bin starts. The explorer
draws them for the digits that occur.

In the Radix Sort view, each pass ranks a working copy of the keys by one
digit and moves the keys to those ranks. In a pair sort, each value moves
with its key. The passes start at ``begin_bit`` and move up four bits at a
time. The last pass is narrower when the range is not a multiple of four
bits. Keys with equal digits keep their order in every pass, so a later
pass keeps the order of the earlier passes among the keys that are equal in
its own digit. In descending order, each pass puts the largest digit first
and still keeps tied keys in their order.

The four bits per pass are fixed. The stages follow the keys from thread to
thread and leave out CUB's instructions and scratch.

With striped output, the last pass returns sorted position ``p`` to thread
``p % T``, item ``p // T``, on a block of ``T`` threads. The passes before
it use blocked order.

Bit ranges
----------

``[begin_bit, end_bit)`` names the bits from ``begin_bit`` to
``end_bit - 1``. The explorer's keys are unsigned, so these are the bits of
the key itself. A sort on some of the bits orders the keys by those bits.
With one item per thread, ``begin_bit=0`` and ``end_bit=4``, an ascending
sort puts the keys 33, 17 and 1 first because their low four bits are 1.
Keys that are equal in those bits keep their input order, whatever their
other bits are.

:ref:`Radix sorting and ranking <coop-cutlass-radix>` in the API reference
has the rules for signed and floating-point keys and for each control.

Using Radix Sort and Rank in a kernel
-------------------------------------

This kernel runs on one block of 64 threads, launched with
``block=(8, 4, 2)``, and the full example passes it ``Int32`` keys. It sorts
the keys by all their bits. It then sorts the keys by their top four bits,
largest first, with striped output. Each key's value is its position in the
tile. Last, it ranks the keys by the same digit in the same direction and
stores the sixteen bin starts that ``exclusive_digit_prefix`` receives. The
two sorts share one ``TempStorage``. ``_BLOCK`` is ``(8, 4, 2)`` and
``_BINS`` is 16. The full example checks all six outputs on the host.

`View the full example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/radix.py>`__

.. literalinclude:: ../../../../python/cuda_coop_cutlass/examples/radix.py
   :language: python
   :start-after: # docs: start cutlass-radix
   :end-before: # docs: end cutlass-radix
   :dedent: 4

The keys are signed, so the top four bits are those of the key with its
sign bit flipped: negative keys have the digits 0 to 7, and the other keys
have the digits 8 to 15.

:ref:`Radix sorting and ranking <coop-cutlass-radix>` in the API reference
covers key types, bit ranges, the order of the result and
``exclusive_digit_prefix``.

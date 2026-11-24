.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-scan:

Scan
====

A scan gives each item the running result of the items before it in its
group. An inclusive scan includes the item itself, and an exclusive scan
stops before it. Items are ordered thread by thread: all of thread 0's
items, then thread 1's, and so on.

The explorer covers :func:`~cuda.coop.cutlass.exclusive_sum`,
:func:`~cuda.coop.cutlass.inclusive_sum`,
:func:`~cuda.coop.cutlass.exclusive_scan` and
:func:`~cuda.coop.cutlass.inclusive_scan`. :func:`~cuda.coop.cutlass.scan`
computes the same scans with its ``mode`` and ``scan_op`` arguments. The
eight teaching threads represent one block, two warps of four lanes, or four
groups of two lanes.

.. coop-visualization:: scan

   .. only:: html

      .. figure:: scan.svg
         :alt: Blocked values 1 through 16 become exclusive sums; each thread
               receives the sum of earlier threads as its incoming prefix.
         :width: 100%

         Exclusive block sum with two items per thread.

   The sum of all earlier threads' items enters each thread, and that
   thread's results continue from it.

   .. code-block:: text

      Thread            0     1      2      3      4      5      6       7
      Input            1,2   3,4    5,6    7,8   9,10  11,12  13,14   15,16
      Incoming prefix   0     3     10     21     36     55     78      105
      Exclusive output 0,1   3,6   10,15  21,28  36,45  55,66  78,91  105,120

Reading the stages
------------------

Local prefixes include only a thread's own items. The propagation stage
computes what enters each thread from the threads before it. Combining that
incoming prefix with the local sequence gives the final result. ``∅`` means
that nothing enters a thread, as in the first thread of an inclusive scan.

The three block algorithms compute the same scan and return each result to
the same thread. ``raking`` passes prefixes through shared-memory segments.
``raking_memoize`` keeps partial results in registers to save shared-memory
reads. ``warp_scans`` scans inside each warp and then combines each result
with what the earlier warps produced. It groups the operations differently
from the two raking algorithms, so floating-point results can differ
between them. The explorer draws the same stages for all three. In a
kernel, ``warp_scans`` needs a block size that is a multiple of 32.

With ``valid_items``, the explorer marks the lanes at or past the count
with ``?``. An exclusive scan combines its initial value into every result:
with maximum, an initial value of 10 keeps every result at least 10.
:ref:`Scan <coop-cutlass-scan>` in the API reference has the rules for each
control.

Using Scan in a kernel
----------------------

This CuTe fragment uses the :ref:`kernel-fragment
setup <coop-visualization-kernels>`. Launch with 128 threads and provide at
least ``128 * items_per_thread`` elements for each block. Each block scans its
own tile independently.

.. code-block:: python

   block = coop.this_block()
   offset = block_index * 128 * items_per_thread
   values = coop.load(
       block, source, items_per_thread=items_per_thread, offset=offset
   )
   prefixes = coop.inclusive_sum(block, values, algorithm="raking_memoize")
   coop.store(block, destination, prefixes, offset=offset)

For an exclusive maximum that starts from 10, replace the scan call with:

.. code-block:: python

   prefixes = coop.exclusive_scan(
       block, values, scan_op="max", initial_value=10
   )

This tested kernel scans the first five lanes of each group of eight lanes,
and every lane writes the sum of those five. ``operator.add`` selects the
built-in sum.

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
   :language: python
   :start-after: # cutlass-exclusive-scan-example-begin
   :end-before: # cutlass-exclusive-scan-example-end
   :dedent: 4

:ref:`Scan <coop-cutlass-scan>` in the API reference covers initial values,
aggregate outputs and which groups take which controls.

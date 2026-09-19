.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-scan:

Scan
====

A scan gives each item the aggregate of earlier items in its group. An
inclusive scan includes the current item; an exclusive scan stops before it.
The order is blocked: all of thread zero's items, then thread one's items,
and so on. Scans return a new scalar or ``ThreadData`` without modifying the
input.

The explorer covers :func:`cuda.coop.exclusive_sum`,
:func:`cuda.coop.inclusive_sum`, :func:`cuda.coop.exclusive_scan`, and
:func:`cuda.coop.inclusive_scan`. :func:`cuda.coop.scan` also exposes these
semantics through its ``mode`` and ``scan_op`` arguments. The eight teaching
threads represent one block, two four-lane physical warps, or four two-lane
logical warps. **Physical CUDA warps have 32 lanes.**

.. coop-visualization:: scan

   .. only:: html

      .. figure:: scan.svg
         :alt: Blocked values 1 through 16 become exclusive sums; each thread receives the sum of earlier threads as its incoming prefix.
         :width: 100%

         Exclusive block sum with two items per thread.

   Local sums establish the prefix entering each thread. Each thread then
   applies that prefix to its own ordered items.

   .. code-block:: text

      Thread            0     1      2      3      4      5      6       7
      Input            1,2   3,4    5,6    7,8   9,10  11,12  13,14   15,16
      Incoming prefix   0     3     10     21     36     55     78      105
      Exclusive output 0,1   3,6   10,15  21,28  36,45  55,66  78,91  105,120

Reading the stages
------------------

Local prefixes include only a thread's own items. The propagation stage
computes what enters each thread from preceding threads. Combining that
incoming prefix with the local sequence produces the final inclusive or
exclusive result. ``∅`` means there is no preceding value, as at the start
of an inclusive scan without an added prefix.

Block Scan offers ``raking``, ``raking_memoize``, and ``warp_scans``.
Raking uses shared-memory segments to propagate prefixes. The memoized form
retains partials in registers to reduce shared-memory reads. Warp scans
compute local warp prefixes and propagate preceding warp aggregates. These
algorithms preserve the same sequence and result ownership; the picture
shows the mathematical dependencies, not an instruction trace. Executable
blocks selecting ``warp_scans`` must contain a multiple of 32 threads.

Block scans accept scalars or several items per thread. Physical- and
logical-warp scans accept one scalar per lane. The qualified warp API also
accepts ``valid_items``: every lane participates, but only the first
``valid_items`` lanes have defined scan results. The explorer marks other
results with ``?``.

Exclusive sums begin at zero. A generic exclusive scan accepts
``initial_value``; a non-sum operator requires it unless a qualified block
prefix callback supplies the prefix. Inclusive scans reject
``initial_value``. The initial value is an operand: for maximum, an initial
value of 10 keeps every output at least 10.

The qualified ``aggregate_output`` returns the input aggregate to every
member, excluding the initial prefix. Prefix callbacks are block-only and
cannot be combined with ``initial_value`` or ``aggregate_output``. The
callback receives the input aggregate and returns the prefix to apply. The
explorer's stateless callback returns ``aggregate + 7``; its stateful example
returns the previous running value of 10 and updates that value with the tile
aggregate.

Using Scan in a kernel
----------------------

This fragment runs inside a Numba-CUDA-MLIR kernel with ``cuda`` imported from
``numba_cuda_mlir``, ``numpy as np``, and ``cuda.coop as coop``. Launch with
128 threads and provide at least 256 elements for each block. Each block
scans its own tile independently.

.. code-block:: python

   block = coop.this_block()
   values = coop.ThreadData(2, dtype=np.int32)
   offset = cuda.blockIdx.x * 256
   coop.load(block, source, values, offset=offset)
   prefixes = coop.inclusive_sum(block, values, algorithm="raking_memoize")
   coop.store(block, output, prefixes, offset=offset)

To compute an exclusive maximum with a prefix, replace the scan call with:

.. code-block:: python

   prefixes = coop.exclusive_scan(
       block, values, scan_op="max", initial_value=np.int32(10)
   )

Custom binary operators use ``scan_op`` through the qualified namespace.
This scalar kernel illustrates an inclusive maximum; launch one block whose
size matches the input:

.. code-block:: python

   from numba_cuda_mlir import cuda
   import cuda.coop.numba_mlir as coop

   @cuda.jit(device=True)
   def maximum(left, right):
       return left if left > right else right

   @cuda.jit
   def running_maximum(source, output):
       thread = cuda.threadIdx.x
       output[thread] = coop.inclusive_scan(
           coop.this_block(), source[thread], scan_op=maximum
       )

A stateless prefix callback uses the qualified ``prefix_op`` argument. Define
this function at module scope, then pass it inside the kernel:

.. code-block:: python

   @cuda.jit(device=True)
   def prefix_after_aggregate(aggregate):
       return aggregate + 7

   # Inside a kernel, with an int32 scalar value in each thread:
   prefix = coop.exclusive_sum(
       coop.this_block(), value, prefix_op=prefix_after_aggregate
   )

For consecutive tiles, a ``StatefulFunction`` carries a running prefix. This
kernel scans two tiles of 128 ``int64`` values with an initial prefix of 10.
Each thread initializes its state cell identically. Only thread zero's final
state is authoritative.

.. code-block:: python

   from numba_cuda_mlir import types

   @cuda.jit(device=True)
   def running_prefix(state, aggregate):
       previous = state[0]
       state[0] = previous + aggregate
       return previous

   prefix_callback = coop.StatefulFunction(
       running_prefix, types.int64, name="running_prefix"
   )

   @cuda.jit
   def scan_two_tiles(source, output, final_state):
       thread = cuda.threadIdx.x
       state = coop.ThreadData(1, dtype=types.int64)
       state[0] = 10
       for tile in range(2):
           index = tile * 128 + thread
           output[index] = coop.exclusive_sum(
               coop.this_block(), source[index], state,
               prefix_op=prefix_callback,
           )
       if thread == 0:
           final_state[0] = state[0]

Keep the default synchronization when reusing scan scratch between calls.
See :doc:`../../coop_api` for the qualified callback and aggregate-output
contracts, and the :doc:`../programming_guide` for backend activation.

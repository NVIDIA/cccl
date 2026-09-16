.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-reduce:

Reduce
======

:func:`cuda.coop.reduce` combines a group's values into one aggregate.
:func:`cuda.coop.sum` is the sum specialization. Each thread can contribute a
scalar or several items, but the return value is a scalar. ``broadcast``
determines which threads may use it.

The explorer uses eight teaching threads. It shows four lanes per physical
warp and two per logical warp; **physical CUDA warps have 32 lanes**. Group
choices also cover a single thread, groups of warps within a block, and a
cluster containing two illustrative blocks.

.. coop-visualization:: reduce

   .. only:: html

      .. figure:: reduce.svg
         :alt: Eight threads fold pairs of input values into partial sums, then reduce to 136 at thread zero; other return values are undefined.
         :width: 100%

         Block sum with two items per thread and ``broadcast=False``.

   Each thread first adds its own pair. The group combines those partials;
   only thread zero has a defined return value in this example.

   .. code-block:: text

      Thread          0     1     2     3     4      5      6      7
      Input          1,2   3,4   5,6   7,8  9,10  11,12  13,14  15,16
      Local sum       3     7    11    15    19     23     27     31
      Return value   136    ?     ?     ?     ?      ?      ?      ?

Reading the stages
------------------

The local fold produces one contribution per thread. The combine row shows
partial aggregates within each selected group. The final row distinguishes
an aggregate available to every member from a return defined only at group
rank zero. ``?`` means that the program must not read that return value.

The default group implementation supports built-in operators across the
thread hierarchy. With ``broadcast=False``, a block can instead select
``raking_commutative_only``, ``raking``, or ``warp_reductions``. The first
requires a commutative operator; the explorer's sum and maximum satisfy that
requirement. Custom callbacks cannot select ``raking_commutative_only``
because the planner cannot establish their commutativity. Raking combines
contributions through shared-memory segments.
Warp reductions combine warp partials before producing the block result.
These rows illustrate legal combinations, rather than exact instruction
schedules. Floating-point rounding can change when the combination order
changes.

``valid_items`` counts contributing group members, starting at rank zero.
It is available for scalar block, physical-warp, and logical-warp reductions
with ``broadcast=False``. Every group member still participates. The
explorer offers this choice only for one item per thread.

A custom operator uses the qualified ``cuda.coop.numba_mlir`` namespace.
It must be associative and is supported through the CUB block or warp path,
with a result defined only at the group root. Custom warp reductions accept
one scalar per lane. The common namespace accepts
built-in names such as ``"sum"``, ``"max"``, ``"min"``, ``"multiplies"``,
and the integer bitwise operators.

Using Reduce in a kernel
------------------------

This fragment runs inside a Numba-CUDA-MLIR kernel with ``cuda`` imported from
``numba_cuda_mlir``, ``numpy as np``, and ``cuda.coop as coop``. Launch with
128 threads and supply at least 256 input elements for each block.

.. code-block:: python

   block = coop.this_block()
   values = coop.ThreadData(2, dtype=np.int32)
   coop.load(block, source, values, offset=cuda.blockIdx.x * 256)
   total = coop.sum(block, values, broadcast=False, algorithm="raking")
   if cuda.threadIdx.x == 0:
       output[cuda.blockIdx.x] = total

The input ``values`` is unchanged. Omit ``algorithm`` and use the default
``broadcast=True`` when every group member needs the aggregate. For one 128-thread block,
this fragment gives every member of each two-warp group the same sum:

.. code-block:: python

   group = coop.this_block().group_by(2)
   total = coop.sum(group, source[cuda.threadIdx.x])
   output[cuda.threadIdx.x] = total

With 128 threads, this creates two groups of 64 threads. A logical warp uses
``coop.this_warp().group_by(8)`` instead, yielding four groups of eight lanes
inside each physical warp. All these groups have separate aggregates.

For the explorer's custom-maximum choice, use a device callback and the
qualified API. Launch this kernel with one block whose size matches the input:

.. code-block:: python

   from numba_cuda_mlir import cuda
   import cuda.coop.numba_mlir as qualified_coop

   @cuda.jit(device=True)
   def maximum(left, right):
       return left if left > right else right

   @cuda.jit
   def block_maximum(source, output):
       thread = cuda.threadIdx.x
       result = qualified_coop.reduce(
           qualified_coop.this_block(),
           source[thread],
           binary_op=maximum,
           broadcast=False,
       )
       if thread == 0:
           output[0] = result

Cluster reduction requires compute capability 9.0 or newer and a cluster
launch. The following kernel and launch use two blocks of 32 threads; all
64 output elements receive the same sum. ``source`` and ``output`` are device
arrays with at least 64 elements.

.. code-block:: python

   from numba_cuda_mlir import cuda
   import cuda.coop as coop

   @cuda.jit
   def cluster_sum(source, output):
       thread = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
       output[thread] = coop.sum(coop.this_cluster(), source[thread])

   cluster_sum.configure(
       (2, 1, 1), (32, 1, 1), cluster=(2, 1, 1)
   )(source, output)

Grid reduction is not supported in this backend. See
:func:`cuda.coop.reduce` and the :doc:`../programming_guide` for the full
operation and group contracts.

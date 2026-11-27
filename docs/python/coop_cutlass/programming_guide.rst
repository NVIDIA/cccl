.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-cutlass:
.. _cuda.coop.cutlass.programming_guide:
.. _cuda-coop-cutlass-cute-dsl-integration:

Programming Guide
=================

This guide starts with a complete program and explains its parts. It then
runs that program on more than one block, shows how primitives reach the
values of CuTe tensors, lists the operations, and ends with how to check and
tune a kernel. :doc:`concepts` introduces the ideas the primitives share,
and the :doc:`API reference <api>` documents every primitive.

.. _coop-cutlass-load-store:

A first program
---------------

This program computes the running sum of 512 integers with one block of 128
threads. Each thread loads four items, the block computes the running sum
across all of them, and each thread stores its four results:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/prefix_sum.py
   :language: python

Run it from the repository root:

.. code-block:: bash

   python python/cuda_coop_cutlass/examples/prefix_sum.py

The kernel has four lines. Each one shows something that
:doc:`Programming concepts <concepts>` explains:

* ``coop.this_block()`` is a :ref:`thread group <coop-concepts-groups>`, the
  128 threads that make each call together.
* ``coop.load`` returns each thread's four items as a
  :ref:`ThreadData <coop-concepts-payloads>`, and ``coop.store`` writes one
  to memory.
* ``coop.inclusive_sum(block, values)`` is a
  :ref:`single-phase call <coop-concepts-single-phase>`. It takes the block
  size from the launch, and the element type and item count from ``values``.
* ``coop.inclusive_sum`` also needs scratch space in shared memory, and the
  compiler provides it. A kernel that wants to size or share that space
  passes a :ref:`TempStorage <coop-concepts-storage>`.

On the host side:

* ``make_ptr(cutlass.Int32, ...)`` gives each pointer its element type. That
  is how ``load`` knows the items are ``Int32``. ``store`` takes the type
  from the ``ThreadData`` it writes, so its pointer must have the same type.
  A pointer to unsigned integers reaches the kernel as the signed type, so
  for ``Uint32`` data pass ``dtype=cutlass.Uint32`` to
  :func:`~cuda.coop.cutlass.load` and :func:`~cuda.coop.cutlass.store`.
* ``items_per_thread`` is declared ``cutlass.Constexpr`` in the kernel and
  in the launcher, so its value is known when the kernel is compiled.
* ``kernel.launch(grid=1, block=128)`` sets the block size.

The first call to ``launch`` compiles the kernel. Each later call runs the
Python code of the launcher and the kernel again, even when CuTe then reuses
the compiled kernel.
:ref:`Checking and tuning a kernel <coop-cutlass-checking>` shows how to
compile once and launch many times.

.. _coop-cutlass-blocks:

More than one block
-------------------

This kernel runs the scan of the first program on a grid of blocks, where
each block computes the running sum of its own :term:`tile`:

:download:`Download the full example and test
<../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py>`

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
   :language: python
   :start-after: # cutlass-concept-blocks-begin
   :end-before: # cutlass-concept-blocks-end

Each block is a group of its own. Its :term:`ranks <rank>` start at 0, and
its scan starts again with its first item. ``load`` and ``store`` never add
the block index, so blocks that pass the same ``offset`` work on the same
tile. This kernel passes an ``offset`` computed from
``cute.arch.block_idx()``: the block's index along x, times the
``128 * items_per_thread`` elements of one tile.

The test launches the kernel with ``grid=3``, ``block=128`` and four items
per thread. The three blocks work on elements 0 to 511, 512 to 1,023 and
1,024 to 1,535. :ref:`Thread groups <coop-thread-groups>` in the API
reference has the rules that hold for every group.

.. _coop-cutlass-cute-tensors:

Working with CuTe tensors
-------------------------

Where a tensor keeps its values decides how a primitive reaches them:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - The values are in
     - A primitive reaches them through
   * - Global or shared memory, behind a CuTe pointer or tensor
     - :func:`~cuda.coop.cutlass.load` and
       :func:`~cuda.coop.cutlass.store`
   * - A CuTe register tensor, made with ``cute.make_rmem_tensor``
     - :meth:`~cuda.coop.cutlass.ThreadData.from_register_tensor` and
       :meth:`~cuda.coop.cutlass.ThreadData.to_register_tensor`
   * - A ``TensorSSA``, such as the result of ``tensor.load()``
     - :meth:`~cuda.coop.cutlass.ThreadData.from_vector` and
       :meth:`~cuda.coop.cutlass.ThreadData.to_tensor_ssa`

``load`` takes its source, and ``store`` its destination, as a CuTe pointer
or as a tensor whose shape and strides are Python values and form a compact
layout. ``cute.make_tensor(pointer, cute.make_layout(n))`` is such a tensor.
So are a shared tensor from CuTe's ``SmemAllocator`` and a tensor from
:meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>`. A
register tensor is rejected there when the kernel is compiled. The entry
for :func:`~cuda.coop.cutlass.load` lists every form and says when the size
of the memory is checked.

Values that are already in registers become a :term:`payload` through the
conversions of :ref:`ThreadData <coop-thread-data>`. Each conversion copies.

:doc:`Advanced topics <advanced>` covers kernels that run native CuTe code
next to the primitives:
:ref:`staging a tile in shared memory <coop-shared-load-store>` with a
native CuTe copy,
:ref:`typed workspaces <coop-cutlass-typed-workspaces>` that native code and
primitives use in turn, and
:ref:`register loads that another object owns <coop-cutlass-producer-interop>`.

.. _coop-cutlass-operations:

Operations
----------

The API reference has one section for each family of primitives, and the
visualization pages draw what the primitives do:

.. list-table::
   :header-rows: 1
   :widths: 20 30 32 18

   * - Family
     - Functions
     - What it does
     - Visualization
   * - :ref:`Load & Store <memory-operations>`
     - :func:`~cuda.coop.cutlass.load`,
       :func:`~cuda.coop.cutlass.store`
     - Moves a group's tile between memory and registers.
     - :doc:`Load <visualizations/load>`,
       :doc:`Store <visualizations/store>`
   * - :ref:`Reduction <coop-cutlass-reduce>`
     - :func:`~cuda.coop.cutlass.reduce`,
       :func:`~cuda.coop.cutlass.sum`,
       :func:`~cuda.coop.cutlass.reduce_batched`
     - Combines the values of a group's threads into one result.
       ``reduce_batched`` gives one result for each item of a
       :term:`lane`'s payload.
     - :doc:`Reduce <visualizations/reduce>`,
       :doc:`Batched Warp Reduction <visualizations/reduce-batched>`
   * - :ref:`Scan <coop-cutlass-scan>`
     - :func:`~cuda.coop.cutlass.scan`,
       :func:`~cuda.coop.cutlass.exclusive_sum`,
       :func:`~cuda.coop.cutlass.inclusive_sum`,
       :func:`~cuda.coop.cutlass.exclusive_scan`,
       :func:`~cuda.coop.cutlass.inclusive_scan`
     - Gives each item the running result of the items before it.
     - :doc:`Scan <visualizations/scan>`
   * - :ref:`Data rearrangement <coop-cutlass-exchange>`
     - :func:`~cuda.coop.cutlass.exchange`,
       :func:`~cuda.coop.cutlass.shuffle`
     - Moves values between the threads of a group.
     - :doc:`Exchange <visualizations/exchange>`,
       :doc:`Shuffle <visualizations/shuffle>`
   * - :ref:`Comparison sorting <coop-cutlass-merge-sort>`
     - :func:`~cuda.coop.cutlass.merge_sort_keys`,
       :func:`~cuda.coop.cutlass.merge_sort_pairs`
     - Puts the keys held by a group's threads in order.
     - :doc:`Merge Sort <visualizations/merge-sort>`
   * - :ref:`Radix sorting and ranking <coop-cutlass-radix>`
     - :func:`~cuda.coop.cutlass.radix_sort_keys`,
       :func:`~cuda.coop.cutlass.radix_sort_pairs`,
       :func:`~cuda.coop.cutlass.radix_rank_keys`
     - Orders a block's keys by their bits, one digit at a time, or ranks
       them by one digit.
     - :doc:`Radix Sort and Rank <visualizations/radix>`
   * - :ref:`Top-k selection <coop-cutlass-topk>`
     - :func:`~cuda.coop.cutlass.topk_min_keys`,
       :func:`~cuda.coop.cutlass.topk_max_keys`,
       :func:`~cuda.coop.cutlass.topk_min_pairs`,
       :func:`~cuda.coop.cutlass.topk_max_pairs`
     - Selects the smallest or largest keys of a block's tile.
     - :doc:`TopK <visualizations/topk>`
   * - :ref:`Neighbor comparisons <coop-cutlass-neighbors>`
     - :func:`~cuda.coop.cutlass.adjacent_difference`,
       :func:`~cuda.coop.cutlass.discontinuity`
     - Compares each item of a block's tile with its neighbor.
     - :doc:`Adjacent Difference <visualizations/adjacent-difference>`,
       :doc:`Discontinuity <visualizations/discontinuity>`
   * - :ref:`Histogram <coop-cutlass-histogram>`
     - :func:`~cuda.coop.cutlass.histogram`,
       :func:`~cuda.coop.cutlass.histogram_init`,
       :func:`~cuda.coop.cutlass.histogram_composite`
     - Counts one tile or accumulates several tiles into shared bins.
     - :doc:`Histogram <visualizations/histogram>`
   * - :ref:`Run Length Decode <coop-cutlass-run-length>`
     - :func:`~cuda.coop.cutlass.run_length_decode`,
       :func:`~cuda.coop.cutlass.run_length_decode_into`
     - Repeats each run's value as many times as its length.
     - :doc:`Run Length Decode <visualizations/run-length-decode>`
   * - :ref:`Compiled device operators <coop-cutlass-device-operators>`
     - :func:`~cuda.coop.cutlass.compile_operator`
     - Lets ``reduce`` and the scans combine values with a function you
       write.
     -

.. _coop-cutlass-checking:

Checking and tuning a kernel
----------------------------

Check a kernel's results against a reference computed on the host before you
time it. Check which thread holds each value as well as the values. Cover
the cases your application uses: a partial tile, an offset other than zero,
more than one warp or group of lanes, and scratch that is reused. Compare
only the results that a primitive defines. Each entry in the API reference
says which threads and which items those are.

Keep compilation out of the time you measure. ``cute.compile`` takes the
``@cute.jit`` function that launches the kernel, followed by that function's
arguments, and returns a callable that you keep. Call it for each launch
with the arguments that are not declared ``cutlass.Constexpr``. The
:ref:`debugger walkthrough <cuda.coop.cutlass.debugger_walkthrough>`
compiles once and launches twice. The first compilation generates the C++
behind the kernel's primitives, compiles it with NVRTC and links it into the
kernel. Another ``cute.compile`` of the same kernel takes the compiled C++
from the package's cache on disk, then compiles and links the kernel again.
Synchronize after the launches you measure and before you read the clock, as
the examples do with ``driver.cuCtxSynchronize()``.

To read the generated C++, set ``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR`` before
the kernel is compiled. :doc:`configuration` describes the variable. Judge
inlining, barriers, shared memory and register use from the final linked
cubin, which CuTe keeps when ``CUTE_DSL_KEEP=cubin`` is set. CuTe writes the
cubin to the current directory, or to the one that ``CUTE_DSL_DUMP_DIR``
names. Set the two CuTe variables before the program starts.
``cuobjdump --dump-sass`` lists the cubin's instructions, where ``LDS`` and
``STS`` are the shared-memory accesses.
``cuobjdump --dump-resource-usage`` lists its registers and static shared
memory. That report can leave out a primitive's scratch, and
:ref:`TempStorage <coop-temp-storage>` shows how to measure it in the
kernel. The link can still inline the code of a primitive into the kernel,
so the generated source and the PTX from before the link show an earlier
stage. Run Compute Sanitizer's race check after you change how a kernel
reuses scratch or where it synchronizes.

More items per thread can raise register use, and more scratch can lower the
number of blocks that fit on a multiprocessor at the same time. Look at the
compiled kernel's resource use as well as its run time.

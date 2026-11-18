.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualizations:

Visualizations
==============

Each page below has an explorer for one primitive or a few related ones.
Change the settings, step through the stages, and select a value to follow
it. All the pictures are simplified in the same way. They draw eight
teaching threads or fewer, and a warp as four lanes where a real warp has
32. They show which values move where, in stages drawn for teaching. They
say nothing about timing or performance.

.. toctree::
   :maxdepth: 1

   load
   store
   exchange
   shuffle
   reduce
   reduce-batched
   scan
   adjacent-difference
   discontinuity
   histogram
   run-length-decode
   merge-sort
   radix
   topk

.. _coop-visualization-kernels:

Using the kernel fragments
--------------------------

The Load, Store, Exchange, Shuffle, Reduce and Scan pages each show a
fragment of a kernel. A fragment goes into a kernel like this one:

.. code-block:: python

   import cutlass
   from cutlass import cute

   import cuda.coop.cutlass as coop


   @cute.kernel
   def kernel(
       source: cute.Pointer,
       destination: cute.Pointer,
       items_per_thread: cutlass.Constexpr,
   ):
       block_index = cute.arch.block_idx()[0]
       thread_rank = cute.arch.thread_idx()[0]
       # The fragment goes here.


   @cute.jit
   def launch(
       source: cute.Pointer,
       destination: cute.Pointer,
       items_per_thread: cutlass.Constexpr,
   ):
       kernel(source, destination, items_per_thread).launch(grid=4, block=128)

* The grid and the blocks have one dimension, so ``block_index`` numbers
  the blocks and ``thread_rank`` is the thread's rank in its block. Every
  fragment is written for blocks of 128 threads.
* ``items_per_thread`` is declared ``cutlass.Constexpr`` in the kernel and
  in the launcher, so its value is known when the kernel is compiled.
* The Reduce and Scan fragments call the second pointer ``output``.
* ``load`` and ``store`` take a pointer as it is, or a tensor in one of the
  forms listed for ``source`` of :func:`cuda.coop.cutlass.load`.
* The Shuffle and Reduce fragments index memory, as in
  ``output[block_index] = total``. The tested kernels on the Shuffle and
  Scan pages index a tensor made from the pointer with
  ``cute.make_tensor(pointer, cute.make_layout(n))``, where ``n`` is the
  number of elements.
* A loop over the items of a payload uses ``cutlass.range_constexpr``, as
  in the Shuffle fragment.

:ref:`A first program <coop-cutlass-load-store>` and
:ref:`More than one block <coop-cutlass-blocks>` in the Programming Guide
explain whole kernels and their launches, and
:ref:`Working with CuTe tensors <coop-cutlass-cute-tensors>` covers pointers
and tensors. :doc:`Executable examples <../examples>` lists complete
programs that check their own output.

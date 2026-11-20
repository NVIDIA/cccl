.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualizations:

Visualizations
==============

Explore how ``cuda.coop.cutlass`` primitives move, combine, compare, count, decode,
order, and choose values. Change the settings, step through the stages, and
select a value to follow its ownership. These diagrams show data movement;
their timing and geometry do not predict GPU performance.

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
---------------------------

The fragments use ``import cuda.coop.cutlass as coop`` with CuTe kernels.
They assume one-dimensional blocks and an ``items_per_thread`` kernel
argument declared ``cutlass.Constexpr`` on both kernel and launcher. Use
``cutlass.range_constexpr`` for loops that index payload slots.

Inside ``@cute.kernel``, obtain ``block_index`` from
``cute.arch.block_idx()[0]`` and ``thread_rank`` from
``cute.arch.thread_idx()[0]``. Pass a supported CuTe tensor or pointer-backed
view to Load/Store. For scalar indexing, construct a tensor with the required
layout. The :doc:`Programming Guide <../programming_guide>` and
:doc:`examples <../examples>` provide complete launch and memory examples.

The diagrams describe the supported contracts. They simplify thread counts
and intermediate stages for teaching; they do not predict instruction timing,
transaction counts, or GPU performance.

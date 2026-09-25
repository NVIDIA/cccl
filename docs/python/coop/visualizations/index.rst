.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualizations:

Visualizations
==============

Explore how ``cuda.coop`` primitives move, combine, compare, count, decode, and order values.
Change the settings, step through the stages, and select a value to follow
its ownership. These diagrams show data movement; their timing and geometry
do not predict GPU performance.

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

The common-API fragments on the Load, Store, Exchange, Shuffle, Reduce,
and Scan pages run under either integration. They use
``from cuda import coop`` and ``import numpy as np``. Register the backend on the host before compiling;
see :ref:`backend registration <coop-backend-registration>`. The fragments
assume one-dimensional blocks and use these local index names:

.. list-table:: DSL setup
   :header-rows: 1

   * - Kernel compiler
     - Imports and decorator
     - ``block_index``
     - ``thread_rank``
   * - Numba-CUDA-MLIR
     - ``from numba_cuda_mlir import cuda``; ``@cuda.jit``
     - ``cuda.blockIdx.x``
     - ``cuda.threadIdx.x``
   * - CUTLASS / CuTe DSL
     - ``from cutlass import cute``; ``@cute.kernel``
     - ``cute.arch.block_idx()[0]``
     - ``cute.arch.thread_idx()[0]``

For example, set ``block_index`` from the appropriate intrinsic inside your
kernel before a fragment computes its tile offset. Numba examples index
device arrays directly. In CuTe, pass global-memory pointers to Load/Store;
for scalar indexing, wrap them in ``cute.make_tensor(pointer,
cute.make_layout(element_count))``. The programming guides provide complete
:doc:`Numba <../programming_guide>` and :doc:`CuTe <../../coop_cutlass>`
launch and memory examples.

The diagrams describe the common contracts implemented by both
Numba-CUDA-MLIR and CUTLASS. Qualified controls are identified on each page.
Custom device operators and Scan prefix callbacks are Numba-qualified
extensions. CuTe supports built-in operators and qualified register-payload
conversions; see :ref:`backend coverage <coop-backends>` and the programming
guides for their respective extensions.

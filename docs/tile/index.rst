.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda-coop-tile:

CCCL Tile Libraries
===================

An exclusive prefix sum turns item counts into starting offsets. Load one
partition of the counts, scan it, and store the result at the same tile
coordinate. Here is the kernel in each language.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../python/cuda_coop_tile/examples/block_scan.py
         :language: python
         :start-at: import cuda.tile as ct
         :end-before: def launch_block_exclusive_sum

   .. tab-item:: C++

      .. literalinclude:: ../../tile/cpp/examples/block_scan.cu
         :language: cpp
         :start-at: namespace ct
         :end-before: bool check

If a 256-element partition begins with counts ``[2, 3, 5, 7]``, its
offsets begin with ``[0, 2, 5, 10]``. The next partition starts at zero
again.

* :doc:`Run the Python example <python/programming>` to check two partitions.
  The :doc:`Python installation guide <python/installation>` lists the
  compiler and package prerequisites.
* :doc:`Build a C++ kernel <cpp/programming>` to call ``cuda::coop::tile`` from
  CUDA C++ and check the result. For a standalone CMake installation, start
  with :doc:`cpp/installation`.

Both languages call CUB primitives through the same TileLib. You pass a
logical Tile value. The compiler selects the physical workers and provides
scratch storage. The :doc:`programming model <programming_model>` explains
how blocks partition an array and how to choose the shape of a result.

For a specific operation, open the :doc:`Python reference <python/api>` or
:doc:`C++ reference <cpp/api>`. The :doc:`data-flow diagrams <visualizations/index>`
show where values go during a scan, exchange, sort, or selection.

.. toctree::
   :hidden:
   :maxdepth: 2

   python/index
   cpp/index
   programming_model
   visualizations/index
   glossary
   faq
   interop
   development

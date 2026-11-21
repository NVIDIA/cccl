.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Tile for C++
============

Use ``cuda::coop::tile`` inside a CUDA C++ Tile kernel. This kernel scans
two partitions in place. Each partition gets its own running sum, starting
at zero.

.. literalinclude:: ../../../tile/cpp/examples/block_scan.cu
   :language: cpp
   :start-at: namespace ct
   :end-before: bool check

Follow the :doc:`programming guide <programming>` to build this program
and check the results. The :doc:`API reference <api>` lists operations by
the result you need.

.. toctree::
   :maxdepth: 2

   programming
   installation
   api
   advanced

.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Tile for Python
===============

Use ``cuda.coop.tile`` inside a cuTile Python kernel. Here each block loads
256 counts and turns them into starting offsets with an exclusive sum.

.. literalinclude:: ../../../python/cuda_coop_tile/examples/block_scan.py
   :language: python
   :start-at: import cuda.tile as ct
   :end-before: def launch_block_exclusive_sum

The :doc:`programming` runs this kernel on two blocks and checks where each
prefix starts. See :doc:`installation` for the compiler and Python packages,
or use the :doc:`api` to look up a primitive by the result you need.

.. toctree::
   :maxdepth: 2

   programming
   installation
   api
   advanced

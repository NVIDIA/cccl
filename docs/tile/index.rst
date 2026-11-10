.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda-coop-tile:

CCCL Tile Libraries
===================

CCCL Tile provides cooperative CUB primitives inside Tile kernels. A call
consumes logical Tile values and returns a scan, reduction, rearrangement,
or other result that can participate in the surrounding kernel computation.
The compiler selects physical participation and allocates temporary storage.

Choose the interface for your kernel language:

* :doc:`Python <python/index>`: ``cuda.coop.tile`` inside ``cuda.tile`` kernels.
* :doc:`C++ <cpp/index>`: ``cuda::coop::tile`` inside CUDA C++ Tile kernels.

Both interfaces share one native provider and TileLib. They provide Scan,
Reduce/Sum, Exchange, Shuffle, Adjacent Difference, Discontinuity, Histogram,
Merge Sort, Radix Sort, Radix Rank, TopK, and Run Length Decode. Operations
act independently within each logical block. The language references describe
supported shapes, dtypes, controls, and result forms.

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

The :doc:`programming_model` explains logical shapes and result ownership.
The :doc:`visualizations/index` show how each primitive transforms a Tile.
For an operation with a similar native Tile counterpart, compare its
semantics and output shape in :ref:`coop-tile-native-or-cooperative`.

Building the provider requires separately supplied Tile/SIMT interoperation
SDK inputs compatible with CUDA 13.5. The :doc:`development` guide describes
the build inputs; each language's installation guide covers its consumer.

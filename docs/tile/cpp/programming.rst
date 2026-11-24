.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Programming guide
=================

Compute an exclusive prefix sum of two 256-element tiles. For each element,
the result is the sum of the earlier elements in the same tile.

Read the kernel
---------------

.. literalinclude:: ../../../tile/cpp/examples/block_scan.cu
   :language: cpp
   :start-at: namespace ct
   :end-before: bool check

The span describes all 512 integers in memory. The partition view splits
them into two tiles of 256 integers. Each logical block uses
``ct::bid().x`` to load its own tile and store the result back to the same
partition. ``coop::exclusive_sum(items)`` returns a new Tile of 256 integers.
The compiler chooses the workers and supplies temporary storage.

The host initializes the array with the integers from 0 through 511 and
launches two logical blocks. The second launch argument is the Tile launch
sentinel ``1``.

.. literalinclude:: ../../../tile/cpp/examples/block_scan.cu
   :language: cpp
   :start-at: block_scan<<<
   :end-at: block_scan<<<
   :dedent: 2

.. list-table::
   :header-rows: 1

   * - Partition
     - First four inputs
     - First four outputs
   * - 0
     - 0, 1, 2, 3
     - 0, 0, 1, 3
   * - 1
     - 256, 257, 258, 259
     - 0, 256, 513, 771

The sum starts again at zero in partition 1. A prefix sum over the entire
array would also need the total from partition 0.

Build and run the program
-------------------------

Follow :doc:`installation` to install the C++ package and set up the CUDA
13.5 compiler. On Linux, run these commands from the repository root for
an SM120 GPU. To build outside the checkout, download
:download:`block_scan.cu <../../../tile/cpp/examples/block_scan.cu>` and
:download:`CMakeLists.txt <../../../tile/cpp/examples/CMakeLists.txt>`
into one directory and pass it to ``cmake -S``.

.. code-block:: bash

   cmake -S tile/cpp/examples -B build/block-scan -G Ninja \
       -DCMAKE_PREFIX_PATH=/path/to/tile-install \
       -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc \
       -DCMAKE_CUDA_ARCHITECTURES=120
   cmake --build build/block-scan
   ./build/block-scan/block_scan

The program waits for the kernel, then checks all 512 results against a host
loop that starts at zero for each tile. A successful run prints:

.. code-block:: text

   PASS: both tiles matched all 256 prefix sums

The CMake project imports the installed headers and TileLib. The
``cuda_coop_tile_link`` helper sets the compile options and tracks the
TileLib as a build dependency:

.. literalinclude:: ../../../tile/cpp/examples/CMakeLists.txt
   :language: cmake
   :start-at: cmake_minimum_required

Using other primitives
----------------------

For a one-dimensional Tile, :doc:`sum(input) <primitives/reduce>` returns
the total at every input position. The axis overload ``sum<Axis>(input)``
returns a Tile of the same rank with extent one on that axis. A
:doc:`histogram <primitives/histogram>` returns one counter per bin. A
:doc:`top-k selection <primitives/topk>` returns a tile with the original
capacity and an unordered selected prefix. Use the reference for the
shape and defined positions of each result.

The :doc:`API reference <api>` has the C++ declarations and examples for
each family. :doc:`advanced` covers algorithm policies, Exchange width,
and kernel worker hints. For the same prefix sum in Python, see
:doc:`/tile/python/programming`.

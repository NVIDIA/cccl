.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Programming guide
=================

Include ``<cuda/coop/tile.cuh>`` and call ``cuda::coop::tile`` from a
``__tile_global__`` kernel. The native installation
provides the header, CMake package, and TileLib. CUDA 13.5
with Tile/SIMT interoperation support is required; see :doc:`/tile/cpp/installation`.

.. rubric:: A first kernel: reductions

This kernel reduces 128 half-precision values and 128 bfloat16 values.
It comes from the C++ runtime test used to exercise both element types:

.. literalinclude:: ../../../tile/cpp/test/support/tile_cpp_frontend_abi_sm120.cu
   :language: cpp
   :start-at: namespace ct =
   :end-before: bool check_cuda

The tensor spans describe the device memory; the partition views select
logical tiles. Each ``sum`` returns a tile containing the aggregate at
every position. The stores preserve that repeated result in the output
arrays. Both calls use temporary storage allocated by the compiler.

After allocating and copying the inputs, the host launches one logical
block and checks CUDA errors:

.. literalinclude:: ../../../tile/cpp/test/support/tile_cpp_frontend_abi_sm120.cu
   :language: cpp
   :start-at:   frontend_abi_kernel<<<
   :end-before:   for (size_t index
   :dedent: 2

The trailing ``1`` is the Tile launch sentinel. It does not request one
physical CUDA thread. Download the :download:`complete executable
<../../../tile/cpp/test/support/tile_cpp_frontend_abi_sm120.cu>`
for device allocation, copies, numerical checks, and cleanup.

.. rubric:: Build an installed consumer with CMake

Place the downloaded source and this ``CMakeLists.txt`` in a directory
outside the CCCL checkout:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.25.2)
   project(tile_reduction LANGUAGES CXX CUDA)

   find_package(cuda-coop-tile CONFIG REQUIRED)
   add_executable(tile_reduction tile_cpp_frontend_abi_sm120.cu)
   cuda_coop_tile_link(tile_reduction)

Configure with the native installation prefix and CUDA 13.5 compiler.
This example targets SM120:

.. code-block:: bash

   cmake -S . -B build -G Ninja \
       -DCMAKE_PREFIX_PATH=/path/to/tile-install \
       -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc \
       -DCMAKE_CUDA_ARCHITECTURES=120
   cmake --build build
   ./build/tile_reduction

``cuda_coop_tile_link(target)`` links the imported interface target
``cuda-coop-tile::tile`` and tracks the TileLib as a compilation dependency.
Changing the TileLib rebuilds the CUDA objects that consume it. The target
supplies include paths, the C++20 requirement, ``--enable-tile``,
``--tile-preview=simt``, and the TileIRAS library option. It does not select
or install a compiler.

For Python-based build orchestration, the optional
:doc:`host helper </tile/python/host_api>`
``get_cpp_config().nvcc_options()`` returns compiler arguments as a tuple.
Pass each item as a separate process argument. Flattening the tuple into
an unquoted shell string loses paths containing spaces.

.. rubric:: Algorithms and Exchange topology

C++ infers the element type and logical extent from each input Tile.
Policies select compile-time algorithms. The compiler and provider select
physical participation for ordinary calls. Assuming ``input`` is a
supported Tile:

.. code-block:: cpp

   namespace coop = ::cuda::coop::tile;

   auto prefixes = coop::exclusive_sum(input);
   auto maximum = coop::reduce(input, ::cuda::maximum<>{});
   auto raking_sum = coop::sum(
     input, coop::reduce_policy<coop::reduce_algorithm::raking>{});

Ordinary policies have no thread-count or items-per-thread parameter.
Exchange requires a count because it determines the permutation:
the compiler must find compatible implementations for the other calls
in that kernel.

.. code-block:: cpp

   auto blocked = coop::exchange_striped_to_blocked(
     input, coop::exchange_policy<128>{});
   auto original = coop::exchange_blocked_to_striped(
     blocked, coop::exchange_policy<128>{});

The :doc:`programming model </tile/programming_model>` gives the permutation
formulas. The :doc:`API reference </tile/cpp/api>` lists policies and their accepted
algorithms, dtypes, and Exchange widths.

.. rubric:: Choosing worker warps

To request eight worker warps, place
``[[cutile::hint(0, num_worker_warps_per_cta=8)]]`` immediately before
the kernel's ``__tile_global__`` declaration. The ``0`` selects the
default architecture entry. The following example uses four and eight
worker warps.

The compiler hint chooses the CTA's workers. A collective may use a subset.
Each call describes its logical input and operation; the provider derives
its implementation. Keep the launch sentinel at ``1``.

This kernel has two worker-count specializations with the same logical
sum call. Each reduces all 256 input elements:

.. code-block:: cpp
   :force:

   template <int WorkerWarps>
   [[cutile::hint(0, num_worker_warps_per_cta=WorkerWarps)]]
   __tile_global__ void worker_sum(const int* input, int* output)
   {
     namespace ct = cuda::tiles;
     namespace coop = cuda::coop::tile;
     auto input_span = ct::tensor_span{input, ct::shape<256>{}};
     auto output_span = ct::tensor_span{output, ct::shape<256>{}};
     auto input_view = ct::partition_view{input_span, ct::shape<256>{}};
     auto output_view = ct::partition_view{output_span, ct::shape<256>{}};
     auto values = input_view.load(0);
     auto aggregate = coop::sum(values);
     output_view.store(aggregate, 0);
   }

Launch ``worker_sum<4><<<1, 1>>>(input, output)`` for four warps or
``worker_sum<8><<<1, 1>>>(input, output)`` for eight. Both produce
the same aggregate at every output position.

.. rubric:: Operators and runtime controls

Supported operation objects select built-in provider implementations.
For example, an exclusive product can start at a runtime value:

.. code-block:: cpp

   auto product = coop::exclusive_scan(
     input, ::cuda::std::multiplies<>{}, initial_value);

The operation object's element type must be ``void`` or exactly match the
input Tile dtype. Bitwise operations require integral elements. These
overloads do not compile arbitrary user-defined functors as provider code.

The API includes scalar and supported shape-one Tile controls for operations
such as scan initial values and sorting bit ranges. Their overloads preserve
the input's compile-time shape. Use the documented control type rather than
assuming an arbitrary one-element wrapper will convert successfully.

.. rubric:: Pairs and decoded output

Pair operations return two values that can be unpacked with structured
bindings. Keys and values keep their association through sorting and
selection. TopK defines only its selected prefix, and that prefix is
unordered.

Run Length Decode is available through this C++ interface. Its
``DecodedN`` template parameter controls output capacity independently of
the number of input runs. ``run_length_decode_with_offsets<DecodedN>``
also returns each decoded position's relative offset within its run.
The reference describes length dtypes, window offsets, and output padding.

.. rubric:: Launching several logical blocks

For larger arrays, construct views over the full device-memory extent
and select the current partition with ``ct::bid().x``. Launch one logical
block for each partition. Each cooperative operation remains local to
that block; a reduction or sort across all partitions needs an additional
algorithm.

Validate more than one partition with distinct inputs. A one-block result
cannot detect an indexing mistake that makes several blocks read or write
the same partition. Keep launch-error checks, synchronization, and numerical
comparison in the host consumer while bringing up a new compiler build.

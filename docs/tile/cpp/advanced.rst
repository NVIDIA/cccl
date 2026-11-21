.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

C++ algorithms and worker warps
===============================

A C++ policy selects an algorithm or sets the width of an Exchange
permutation at compile time. These examples use a 256-element integer
Tile named ``input``.

Algorithms and Exchange topology
--------------------------------

.. code-block:: cpp

   namespace coop = ::cuda::coop::tile;

   auto prefixes = coop::exclusive_sum(input);
   auto maximum = coop::reduce(input, ::cuda::maximum<>{});
   auto raking_sum = coop::sum(
     input, coop::reduce_policy<coop::reduce_algorithm::raking>{});

The scan and reductions above use the worker count chosen by the compiler.
For Exchange, the width determines the permutation. These two calls use
the same width and undo one another:

.. code-block:: cpp

   auto blocked = coop::exchange_striped_to_blocked(
     input, coop::exchange_policy<128>{});
   auto original = coop::exchange_blocked_to_striped(
     blocked, coop::exchange_policy<128>{});

The :doc:`programming model </tile/programming_model>` gives the permutation
formulas. The :doc:`API reference </tile/cpp/api>` lists policies and their accepted
algorithms, dtypes, and Exchange widths.

Choosing worker warps
---------------------

To request eight worker warps, place
``[[cutile::hint(0, num_worker_warps_per_cta=8)]]`` immediately before
the kernel's ``__tile_global__`` declaration. The ``0`` selects the
default architecture entry. The following example uses four and eight
worker warps.

The compiler hint chooses the CTA's workers. A collective may use a subset.
Keep the second launch argument at ``1`` for either specialization.

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

Operators and runtime controls
------------------------------

An exclusive product can start at a runtime value:

.. code-block:: cpp

   auto product = coop::exclusive_scan(
     input, ::cuda::std::multiplies<>{}, initial_value);

A typed functor must match the input element type. A transparent functor
such as ``cuda::std::multiplies<>{}`` uses that type. See the
:doc:`scan overloads <primitives/scan>` for supported operators, seeds, and
shape-one Tile controls.

For a native Tile and cooperative reduction side by side, see
:doc:`primitives/reduce`. For a complete kernel that returns decoded values
together with positions within each run, see :doc:`primitives/run_length_decode`.

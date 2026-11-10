.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Tile for C++
============

``cuda::coop::tile`` provides cooperative primitives inside CUDA C++ Tile kernels.
Start with :doc:`installation`, then use the :doc:`programming` to compile,
launch, and check a complete kernel. See :doc:`/tile/programming_model`
for the shared logical Tile model and :doc:`api` for operation contracts.

.. code-block:: cpp

   auto values = input_view.load(ct::bid().x);
   auto prefixes = coop::exclusive_sum(values);
   output_view.store(prefixes, ct::bid().x);

Here ``ct`` names ``cuda::tiles`` and ``coop`` names ``cuda::coop::tile``.
The views each select a 256-element partition.

.. toctree::
   :maxdepth: 2

   installation
   programming
   api

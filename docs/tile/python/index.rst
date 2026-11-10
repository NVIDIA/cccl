.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Tile for Python
===============

``cuda.coop.tile`` provides cooperative primitives inside cuTile Python kernels.
Start with :doc:`installation`, then use the :doc:`programming` to compile,
launch, and check a complete kernel. See :doc:`/tile/programming_model`
for the shared logical Tile model and :doc:`api` for operation contracts.

.. code-block:: python

   values = ct.load(input_array, index=(ct.bid(0),), shape=(256,))
   prefixes = coop.exclusive_sum(values)
   ct.store(output_array, index=(ct.bid(0),), tile=prefixes)

Here ``ct`` names ``cuda.tile`` and ``coop`` names ``cuda.coop.tile``.

.. toctree::
   :maxdepth: 2

   installation
   programming
   api

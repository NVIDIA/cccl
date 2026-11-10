.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Reduction
=========

.. py:currentmodule:: cuda.coop.tile

Reduction combines values along selected Tile axes. Python ``sum`` and
``reduce`` return one aggregate for each combination of retained indices.
For example, summing an ``(8, 128)`` Tile along axis one returns eight sums.
Both operations use CUB and reduce within a tile, not across a launch.

.. py:function:: sum(value, /, axis=None, keepdims=False, *, algorithm=None, methods=None, broadcast=False)
   :no-contents-entry:

   Sum ``value`` along the selected axes.

.. py:function:: reduce(value, /, axis=None, keepdims=False, *, binary_op=None, algorithm=None, methods=None, broadcast=False)
   :no-contents-entry:

   Reduce ``value`` along the selected axes using ``binary_op``.

``value``
   A supported numeric Tile. Accumulation and output keep the input dtype;
   addition of boolean values promotes to ``int32``.
``axis``
   ``None`` reduces all elements. A compile-time integer selects one axis;
   a tuple selects several axes. Negative indices count from the last axis.
   Repeated or out-of-range axes are errors. An empty tuple leaves the values
   unchanged, with boolean addition still promoting to ``int32``.
``keepdims``
   A compile-time boolean. The default removes reduced axes; ``True`` retains
   them with extent one. Reducing all axes returns a rank-zero Tile with shape
   ``()`` by default. A shape ``(1,)`` Tile has one axis and is not rank zero.
``binary_op``
   A compile-time name: ``"sum"`` or ``"+"`` (default), ``"*"``, ``"min"``,
   ``"max"``, ``"&"``, ``"|"``, or ``"^"``. Bitwise operations require an
   integer dtype. User-defined Python callbacks are unsupported.
``algorithm``
   ``"warp_reductions"`` (default), ``"raking"``, or
   ``"raking_commutative_only"``. The last requires a commutative operation.
``methods``
   ``sum`` accepts ``algorithm``. ``reduce`` accepts ``binary_op`` and
   ``algorithm``.
``broadcast``
   A compile-time boolean. ``True`` repeats each aggregate across its reduced
   axes to restore the input shape; ``keepdims`` then has no effect on shape.
   Use this option when the consumer needs an input-shaped result.

For an already loaded ``matrix`` of shape ``(8, 128)`` and ``values`` of shape
``(128,)``:

.. code-block:: python

   import cuda.coop.tile as coop

   rows = coop.sum(matrix, axis=1)                 # (8,)
   gradient = coop.sum(matrix, axis=-1, keepdims=True)  # (8, 1)
   total = coop.sum(values)                        # ()
   total_1d = coop.sum(values, keepdims=True)       # (1,)
   repeated = coop.sum(values, broadcast=True)     # (128,)
   minimum = coop.reduce(matrix, axis=1, keepdims=True, binary_op="min")
   shifted = matrix - minimum                     # (8, 128)

Reduced results also participate in ordinary Tile broadcasting. Subtracting a
row minimum needs ``keepdims=True`` so the retained row axis aligns with the
matrix. There is no need to extract or insert rows around a reduction.

Independent reductions are batched in the provider. A 128-element reduction
supports kernels requesting four, eight, or sixteen worker warps; changing
that worker count does not change the mathematical input shape.

.. rubric:: Example: subtract a Tile's minimum

A reduced scalar broadcasts in an elementwise expression. This complete
example checks all 256 output positions against an independent reference:

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def shift_kernel(source, destination):
       values = ct.load(source, (ct.bid(0),), shape=(256,))
       minimum = coop.reduce(values, binary_op="min")
       ct.store(destination, (ct.bid(0),), values - minimum)

   source = torch.arange(256, device="cuda", dtype=torch.int32) - 19
   destination = torch.empty_like(source)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), shift_kernel, (source, destination))
   assert torch.equal(destination, source - source.min())


Numerical behavior and native Tile comparison
---------------------------------------------

Python axes, result shapes, ``keepdims``, and boolean sum promotion follow
`native Tile sum <https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.sum.html>`_.
Cooperative reductions do not expose native Tile's directed rounding or
flush-to-zero controls. Floating-point accumulation keeps the input type,
including ``float16`` and ``bfloat16``. It does not widen to improve accuracy.
Integer overflow is not detected.

Native and cooperative results can both feed subsequent Tile expressions:

For an already loaded one-dimensional ``values`` Tile:

.. code-block:: python

   import cuda.tile as ct
   import cuda.coop.tile as coop

   total = ct.sum(values)                   # shape ()
   total_1d = ct.sum(values, keepdims=True)  # shape (1,)
   repeated = coop.sum(values, broadcast=True)  # input shape
   shifted = values - ct.max(values)


Floating-point sums and products can differ from native Tile or a serial CPU
reference because the reduction order differs. Compare values with tolerances
suited to the dtype and input range. Sums propagate NaN and combine positive
and negative infinity into NaN. The sign of a zero sum may depend on the
reduction order.

Generic minimum and maximum preserve the existing CUB-style comparison
operators. They retain the first operand on ties or unordered comparisons;
NaN and signed-zero results may depend on operand order. They do not promise
native ``ct.min`` or ``ct.max`` NaN-propagation semantics. Noncontributing
lanes do not supply artificial minimum or maximum values.

Benchmark the complete consumer kernel when choosing between cooperative and
native reductions; neither promises to be faster for every shape or worker
count. See the :ref:`native-versus-cooperative FAQ <coop-tile-native-or-cooperative>`.

.. figure:: /tile/visualizations/reduce.svg
   :alt: Broadcast reduction repeats the aggregate across the output Tile.

   Python ``broadcast=True`` and C++ calls without an axis repeat the aggregate across
   the input extent. Python's default result contains only the aggregates.

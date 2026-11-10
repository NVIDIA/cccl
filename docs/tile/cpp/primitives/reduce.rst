.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Reduction
=========

Reduction combines values within a Tile using CUB. Axis overloads produce
one aggregate for each combination of retained indices; overloads without
an axis repeat the aggregate across a one-dimensional Tile.

``sum<Axis>(input, policy)`` and ``reduce<Axis>(input, operation, policy)``
reduce one compile-time axis of a multidimensional Tile. They preserve
its rank, replacing the reduced extent with one, as native C++ Tile does.
For example, ``coop::sum<1>(matrix)`` maps ``ct::shape<8, 128>`` to
``ct::shape<8, 1>``; ``coop::sum<0>(values)`` maps ``ct::shape<128>`` to
``ct::shape<1>``. This is a one-dimensional result, distinct from Python's
default rank-zero Tile. Axes must be nonnegative and less than the rank.
Compose calls to reduce several axes. C++ boolean inputs remain unsupported.

Calls without an axis retain their one-dimensional broadcast contract.
``sum(input, policy)`` and ``reduce(input, operation, policy)`` accept shape
``ct::shape<N>`` and return that same shape and element type. Every output
position contains the aggregate. C++ calls do not take Python's ``axis``,
``keepdims``, or ``broadcast`` arguments.

For valid C++ Tile shapes, the number of elements reduced per result must
be 1, 2, 4, 8, 16 or a positive multiple of 32. This is ``N`` for calls
without an axis, or the reduced-axis extent for axis overloads.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto total = coop::sum(input);
   auto minimum = coop::reduce(input, cuda::minimum<>{});
   auto shifted = input - minimum;

``reduce_policy<Algorithm>`` defaults to ``reduce_algorithm::warp_reductions``;
``raking`` and ``raking_commutative_only`` are also supported. See the
:doc:`C++ API reference </tile/cpp/api>` for overloads. Both forms use the
batched provider and support a 128-element input with four, eight, or
sixteen kernel worker warps.


Numerical behavior and native Tile comparison
---------------------------------------------

Cooperative reductions do not expose native Tile's directed rounding or
flush-to-zero controls. Floating-point accumulation keeps the input type,
including ``__half`` and ``__nv_bfloat16``. It does not widen to improve accuracy.
Integer overflow is not detected.

Native and cooperative results can both feed subsequent Tile expressions:

.. code-block:: cpp

   namespace ct = cuda::tiles;
   namespace coop = cuda::coop::tile;

   auto total = ct::sum<0>(values);
   auto repeated = coop::sum(values);
   auto shifted = values - ct::reduce_max<0>(values);

The native calls retain axis zero with extent one, returning shape
``ct::shape<1>``. The cooperative sum repeats the total across the input.


Floating-point sums and products can differ from native Tile or a serial CPU
reference because the reduction order differs. Compare values with tolerances
suited to the dtype and input range. Sums propagate NaN and combine positive
and negative infinity into NaN. The sign of a zero sum may depend on the
reduction order.

Generic minimum and maximum preserve the existing CUB-style comparison
operators. They retain the first operand on ties or unordered comparisons;
NaN and signed-zero results may depend on operand order. They do not promise
native ``ct::reduce_min`` or ``ct::reduce_max`` NaN-propagation semantics.
Noncontributing lanes do not supply artificial minimum or maximum values.

Benchmark the complete consumer kernel when choosing between cooperative and
native reductions; neither promises to be faster for every shape or worker
count. See the :ref:`native-versus-cooperative FAQ <coop-tile-native-or-cooperative>`.

.. figure:: /tile/visualizations/reduce.svg
   :alt: Broadcast reduction repeats the aggregate across the output Tile.

   Python ``broadcast=True`` and C++ calls without an axis repeat the aggregate across
   the input extent. Python's default result contains only the aggregates.

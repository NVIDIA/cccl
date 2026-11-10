.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Scan and prefix sums
====================

Scan computes a prefix at each position. An exclusive prefix omits the current
input; an inclusive prefix includes it. For ``[2, 1, 3, 2]``, addition produces
``[0, 2, 3, 6]`` exclusively and ``[2, 3, 6, 8]`` inclusively.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

The input and result are ``cuda::tiles::tile<T, cuda::tiles::shape<N>>``
with the same numeric element type and extent ``N``, which must be a positive
multiple of 32. Accumulation stays in ``T``. Bitwise operations require an
integer type. Exclusive addition defaults to a zero seed; other exclusive
operations require a seed appropriate to the
operation. Physical participation is selected for the logical extent.

``exclusive_sum``, ``inclusive_sum``, ``exclusive_scan``, and ``inclusive_scan``
accept an optional ``scan_policy<Algorithm>``. Its default algorithm is
``scan_algorithm::raking``. ``scan<scan_mode::exclusive>`` and
``scan<scan_mode::inclusive>`` select the mode as a template argument.

Generic operations accept ``cuda::std::plus``, ``multiplies``, ``bit_and``,
``bit_or``, ``bit_xor``, ``cuda::minimum``, or ``cuda::maximum``. Typed functors
must match the Tile element type; transparent functors are also supported.
The generic exclusive form accepts its initial value after the operation;
the inclusive form accepts a prefix in the same position. Both sum forms
accept a prefix immediately after the input. Each value can be a scalar
convertible to the input element type ``T`` or a
``cuda::tiles::tile<T, cuda::tiles::shape<1>>`` with exactly the input dtype.
``scan<Mode>(input, operation, value)`` interprets ``value`` as an exclusive
initial value or an inclusive prefix according to ``Mode``; omitting the
operation selects addition. For the supported commutative operations, the
exclusive initial-value form computes the same mathematical prefix as
Python's ``prefix_value``. Floating-point association can differ.
For an already loaded ``input`` Tile:

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto offsets = coop::exclusive_sum(input);
   auto products = coop::exclusive_scan(input, cuda::std::multiplies<>{}, 1);
   auto maxima = coop::inclusive_scan(input, cuda::maximum<>{});
   auto inclusive = coop::scan<coop::scan_mode::inclusive>(input);


Behavior
--------


With a prefix of 10, the exclusive example becomes ``[10, 12, 13, 16]`` and the
inclusive example becomes ``[12, 13, 16, 18]``. For multiplication, use an
exclusive seed of 1 to obtain the usual product prefix. Choose seeds in the
input dtype; the provider does not widen accumulation.

.. figure:: /tile/visualizations/scan.svg
   :alt: Exclusive and inclusive prefixes with seed values and different operations.

   Prefix boundaries, seeds, multiplication, and maximum for the same input.
   This small array illustrates the operation; it is not a supported launch shape.

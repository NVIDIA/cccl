.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

C++ API reference
=================

Include ``<cuda/coop/tile.cuh>`` and call these functions in a
``__tile_global__`` kernel. ``ct`` below is the header's namespace alias for
``cuda::tiles``; ``cte`` aliases ``cuda::tiles::experimental``.
The :doc:`/tile/cpp/programming` guide covers compilation and launch.

Each entry shows a declaration from the public header and describes its
other overloads. Execution annotations and ``inline`` are omitted; type
requirements are stated beside the declarations. The :download:`header
<../../../tile/cpp/include/cuda/coop/tile.cuh>` contains
the complete overload set, including the operation-specific constraints.

Data Tiles have positive extents and supported numeric
C++ element types: signed and unsigned 8-, 16-, 32-, and 64-bit integers,
``__half``, ``__nv_bfloat16``, ``float``, and ``double``. ``bool`` and
structured types are unsupported. Pair inputs must have equal extents.
For operations other than reductions, the input extent ``N`` must be a
positive multiple of 32. For valid C++ Tile shapes, reductions support
1, 2, 4, 8, 16 or a positive multiple of 32 elements per result: ``N`` for
calls without an axis, or the reduced-axis extent for axis overloads.
The provider must find a physical worker count compatible with every call
in the kernel. See :doc:`/tile/programming_model` for logical shapes and physical
participation.

Policies and options
--------------------

.. cpp:namespace:: cuda::coop::tile

.. cpp:enum-class:: scan_algorithm

   .. cpp:enumerator:: raking

      Raking scan.

   .. cpp:enumerator:: raking_memoize

      Raking scan with memoized partial results.

   .. cpp:enumerator:: warp_scans

      Warp scan composition.

.. cpp:enum-class:: reduce_algorithm

   .. cpp:enumerator:: raking_commutative_only

      Raking reduction for a commutative operation.

   .. cpp:enumerator:: raking

      Raking reduction.

   .. cpp:enumerator:: warp_reductions

      Warp reduction composition.

.. cpp:enum-class:: radix_smem_config

   .. cpp:enumerator:: four_byte

      Four-byte shared-memory banks.

   .. cpp:enumerator:: eight_byte

      Eight-byte shared-memory banks.

.. cpp:enum-class:: scan_mode

   .. cpp:enumerator:: exclusive

      Exclude the current input from its prefix.

   .. cpp:enumerator:: inclusive

      Include the current input in its prefix.

.. cpp:struct:: template<scan_algorithm Algorithm = scan_algorithm::raking> scan_policy
   :no-typesetting:

.. code-block:: cpp

   template <scan_algorithm Algorithm = scan_algorithm::raking>
   struct scan_policy;

Select the scan algorithm. The default is ``raking``.

.. cpp:struct:: template<reduce_algorithm Algorithm = reduce_algorithm::warp_reductions> reduce_policy
   :no-typesetting:

.. code-block:: cpp

   template <reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
   struct reduce_policy;

Select the reduction algorithm. The default is ``warp_reductions``.

.. cpp:struct:: template<size_t BlockThreads> exchange_policy
   :no-typesetting:

.. code-block:: cpp

   template <size_t BlockThreads> struct exchange_policy;

Select the width defining the Exchange permutation. ``BlockThreads`` must be a
multiple of 32 in ``[32, 1024]`` and divide the input extent. The kernel must admit
that width.

.. cpp:struct:: template<size_t RadixBits = 4, bool MemoizeOuterScan = true, scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans, radix_smem_config SmemConfig = radix_smem_config::four_byte> radix_sort_policy
   :no-typesetting:

.. code-block:: cpp

   template <size_t RadixBits = 4, bool MemoizeOuterScan = true,
             scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
             radix_smem_config SmemConfig = radix_smem_config::four_byte>
   struct radix_sort_policy;

Select bits per sorting pass, outer-scan memoization, inner-scan algorithm, and
shared-memory bank width. ``RadixBits`` is in ``[1, 8]``; it does not limit the
total sorted bit interval.
See :ref:`tile-radix-capacity` for resource limits on usable values.

.. cpp:struct:: template<size_t RadixBits = 4, bool Descending = false, bool MemoizeOuterScan = true, scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans, radix_smem_config SmemConfig = radix_smem_config::four_byte> radix_rank_policy
   :no-typesetting:

.. code-block:: cpp

   template <size_t RadixBits = 4, bool Descending = false,
             bool MemoizeOuterScan = true,
             scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
             radix_smem_config SmemConfig = radix_smem_config::four_byte>
   struct radix_rank_policy;

Select radix-digit capacity and direction, plus scan and bank settings.
``RadixBits`` is in ``[1, 16]`` and must cover the requested digit. Ascending rank
is the default.
See :ref:`tile-radix-capacity` for resource limits on usable values.

.. cpp:struct:: template<typename... Options> topk_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct topk_policy;

An empty policy. Use ``topk_policy<>``; supplying any option is a compile-time
error.

.. cpp:struct:: template<typename... Options> histogram_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct histogram_policy;

An empty policy. Use ``histogram_policy<>``; supplying any option is a compile-time
error.

.. cpp:struct:: template<typename... Options> run_length_decode_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct run_length_decode_policy;

An empty policy. Use ``run_length_decode_policy<>``; supplying any option is a
compile-time error.

.. cpp:struct:: template<typename... Options> discontinuity_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct discontinuity_policy;

An empty policy. Use ``discontinuity_policy<>``; supplying any option is a compile-time error.

.. cpp:struct:: template<typename... Options> adjacent_difference_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct adjacent_difference_policy;

An empty policy. Use ``adjacent_difference_policy<>``; supplying any option is a
compile-time error.

.. cpp:struct:: template<typename... Options> shuffle_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct shuffle_policy;

An empty policy. Use ``shuffle_policy<>``; supplying any option is a compile-time
error.

.. cpp:struct:: template<typename... Options> merge_sort_policy
   :no-typesetting:

.. code-block:: cpp

   template <typename... Options> struct merge_sort_policy;

An empty policy. Use ``merge_sort_policy<>``; supplying any option is a compile-time error.

Policies select compile-time options and allocate no scratch storage.
Except for Exchange, they contain no physical thread-count parameter.

Scan and reduction
------------------

.. cpp:function:: template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking> ct::tile<T, ct::shape<N>> exclusive_sum(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N,
             scan_algorithm Algorithm = scan_algorithm::raking>
   ct::tile<T, ct::shape<N>>
   exclusive_sum(ct::tile<T, ct::shape<N>> input,
                 scan_policy<Algorithm> = {}) noexcept;

Return the sum of preceding elements, with zero at position zero. The result
preserves the input shape and dtype. A prefix overload accepts
``exclusive_sum(input, prefix_value, policy = {})``. The prefix supplies the
first result and may be a scalar convertible to ``T`` or a shape-one Tile of
exactly that dtype.

.. cpp:function:: template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking> ct::tile<T, ct::shape<N>> inclusive_sum(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N,
             scan_algorithm Algorithm = scan_algorithm::raking>
   ct::tile<T, ct::shape<N>>
   inclusive_sum(ct::tile<T, ct::shape<N>> input,
                 scan_policy<Algorithm> = {}) noexcept;

Return the sum through each input position, preserving the input shape and dtype.
A prefix overload accepts ``inclusive_sum(input, prefix_value, policy = {})``
and adds the prefix before the first input. Prefix scalar and Tile forms match
``exclusive_sum``.

.. cpp:function:: template <scan_mode Mode = scan_mode::exclusive, typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking> ct::tile<T, ct::shape<N>> scan(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <scan_mode Mode = scan_mode::exclusive, typename T, size_t N,
             scan_algorithm Algorithm = scan_algorithm::raking>
   ct::tile<T, ct::shape<N>> scan(ct::tile<T, ct::shape<N>> input,
                                  scan_policy<Algorithm> policy = {}) noexcept;

Select the mode with ``scan_mode``. The overload without an operation computes
addition. An operation may follow ``input``, followed by an optional value.
``scan<scan_mode::exclusive>(input, operation, value)`` uses that value as an
initial value; ``scan<scan_mode::inclusive>(input, operation, value)`` uses it as
a prefix. ``scan<Mode>(input, value)`` uses addition with the same mode-dependent
meaning. Exclusive non-addition requires a seed. Supported operations and value
types are listed below.

.. cpp:function:: template <typename T, size_t N, typename OpT, scan_algorithm Algorithm = scan_algorithm::raking> ct::tile<T, ct::shape<N>> exclusive_scan(ct::tile<T, ct::shape<N>> input, ::cuda::std::plus<OpT>, scan_policy<Algorithm> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N, typename OpT,
             scan_algorithm Algorithm = scan_algorithm::raking>
   ct::tile<T, ct::shape<N>>
   exclusive_scan(ct::tile<T, ct::shape<N>> input, ::cuda::std::plus<OpT>,
                  scan_policy<Algorithm> policy = {}) noexcept;

Compute an exclusive prefix, preserving shape and dtype. The overload without an
operation computes addition with a zero seed. Addition also accepts a seed
immediately after ``input``. With an operation, supply it after ``input`` and the
seed after the operation. Only addition can omit its seed. Seeds accept a scalar
convertible to ``T`` or ``ct::tile<T, ct::shape<1>>`` with exactly the input dtype.

.. cpp:function:: template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking> ct::tile<T, ct::shape<N>> inclusive_scan(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N,
             scan_algorithm Algorithm = scan_algorithm::raking>
   ct::tile<T, ct::shape<N>>
   inclusive_scan(ct::tile<T, ct::shape<N>> input,
                  scan_policy<Algorithm> policy = {}) noexcept;

Compute an inclusive prefix, preserving shape and dtype. Omitting the operation
selects addition. An operation may follow ``input``, followed by an optional
prefix: ``inclusive_scan(input, operation, prefix_value, policy = {})``. The
addition form may omit the operation: ``inclusive_scan(input, prefix_value,
policy = {})``. Prefixes accept a scalar convertible to ``T`` or a shape-one
Tile with exactly the input dtype. A prefix combines with the first input;
it is not an exclusive initial-value overload.

.. cpp:function:: template <typename T, size_t N, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions> ct::tile<T, ct::shape<N>> sum(ct::tile<T, ct::shape<N>> input, reduce_policy<Algorithm> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N,
             reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
   ct::tile<T, ct::shape<N>> sum(ct::tile<T, ct::shape<N>> input,
                                 reduce_policy<Algorithm> = {}) noexcept;

Return the sum repeated across the input shape, using the input dtype for
accumulation. ``T`` must be a supported numeric type.

.. cpp:function:: template <size_t Axis, typename T, typename Shape, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions> ct::reduction_result_t<ct::tile<T, Shape>, Axis> sum(ct::tile<T, Shape> input, reduce_policy<Algorithm> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <size_t Axis, typename T, typename Shape,
             reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
   ct::reduction_result_t<ct::tile<T, Shape>, Axis>
   sum(ct::tile<T, Shape> input, reduce_policy<Algorithm> = {}) noexcept;

Sum along one compile-time axis and retain that dimension with extent one,
following native C++ Tile conventions. ``Axis`` must be less than the input
rank. For shape ``ct::shape<8, 128>``, ``sum<1>(input)`` returns shape
``ct::shape<8, 1>``. A one-dimensional ``sum<0>(input)`` returns shape
``ct::shape<1>``. Compose calls to reduce several axes; retained dimensions
keep subsequent axis indices unchanged. Accumulation and output keep ``T``;
boolean inputs are unsupported.

.. cpp:function:: template <typename T, size_t N, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions> ct::tile<T, ct::shape<N>> reduce(ct::tile<T, ct::shape<N>> input, reduce_policy<Algorithm> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N,
             reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
   ct::tile<T, ct::shape<N>>
   reduce(ct::tile<T, ct::shape<N>> input,
          reduce_policy<Algorithm> policy = {}) noexcept;

Repeat the aggregate across the input shape. Omitting the operation selects
addition; otherwise an operation follows ``input``. ``T`` must be numeric and a
typed operation must match it. Integer bitwise operations additionally require
integral ``T``.

.. cpp:function:: template <size_t Axis, typename T, typename Shape, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions> ct::reduction_result_t<ct::tile<T, Shape>, Axis> reduce(ct::tile<T, Shape> input, reduce_policy<Algorithm> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <size_t Axis, typename T, typename Shape,
             reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
   ct::reduction_result_t<ct::tile<T, Shape>, Axis>
   reduce(ct::tile<T, Shape> input, reduce_policy<Algorithm> policy = {}) noexcept;

Reduce along ``Axis`` with the same retained-dimension result as ``sum<Axis>``.
An optional operation follows ``input``; for example,
``reduce<1>(input, cuda::minimum<>{})`` computes one minimum per row.
Axes are nonnegative compile-time indices. Tuple axes and Python's ``keepdims``
and ``broadcast`` arguments are not C++ parameters.

Both axis and one-dimensional broadcast overloads use the batched provider.
A 128-element reduction supports four, eight, and sixteen kernel worker warps
without changing the input shape. Floating evaluation order can change with
the selected implementation; bitwise equality is not guaranteed.

Scan and reduction operation overloads accept ``cuda::std::plus<OpT>``,
``cuda::std::multiplies<OpT>``, ``cuda::minimum<OpT>``,
``cuda::maximum<OpT>``, ``cuda::std::bit_and<OpT>``,
``cuda::std::bit_or<OpT>``, and ``cuda::std::bit_xor<OpT>``.
``OpT`` must be ``void`` or exactly the input element type; bitwise
operations require integral elements. These are concrete overloads for
those functors, not an arbitrary callable interface. The optional policy
is always the last argument. Accumulation stays in the input dtype.
See :doc:`/tile/cpp/primitives/scan` and :doc:`/tile/cpp/primitives/reduce` for numerical behavior.

Movement and neighbors
----------------------

.. cpp:function:: template <typename T, size_t N, size_t BlockThreads> ct::tile<T, ct::shape<N>> exchange_striped_to_blocked(ct::tile<T, ct::shape<N>> input, exchange_policy<BlockThreads> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N, size_t BlockThreads>
   ct::tile<T, ct::shape<N>>
   exchange_striped_to_blocked(ct::tile<T, ct::shape<N>> input,
                               exchange_policy<BlockThreads> = {}) noexcept;

For ``B = BlockThreads`` and ``I = N / B``, return ``output[t * I + i] = input[i *
B + t]``. Supply ``exchange_policy<B>{}`` explicitly so ``B`` can be deduced. The
default function argument does not supply a default width.

.. cpp:function:: template <typename T, size_t N, size_t BlockThreads> ct::tile<T, ct::shape<N>> exchange_blocked_to_striped(ct::tile<T, ct::shape<N>> input, exchange_policy<BlockThreads> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N, size_t BlockThreads>
   ct::tile<T, ct::shape<N>>
   exchange_blocked_to_striped(ct::tile<T, ct::shape<N>> input,
                               exchange_policy<BlockThreads> = {}) noexcept;

Return the inverse permutation, ``output[i * B + t] = input[t * I + i]``. Supply
the same explicit ``exchange_policy<B>{}`` as the forward conversion.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> shuffle_up(ct::tile<T, ct::shape<N>> input, shuffle_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> shuffle_up(ct::tile<T, ct::shape<N>> input,
                                        shuffle_policy<> = {}) noexcept;

Read the preceding element and preserve the first input element. Shape and dtype
are unchanged. Multiple items per physical worker are supported.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> shuffle_down(ct::tile<T, ct::shape<N>> input, shuffle_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> shuffle_down(ct::tile<T, ct::shape<N>> input,
                                          shuffle_policy<> = {}) noexcept;

Read the following element and preserve the last input element. Shape and dtype are
unchanged. Multiple items per physical worker are supported.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> shuffle_offset(ct::tile<T, ct::shape<N>> input, int distance = 1, shuffle_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> shuffle_offset(ct::tile<T, ct::shape<N>> input,
                                            int distance = 1,
                                            shuffle_policy<> = {}) noexcept;

Return ``input[i + distance]`` when the source is in range; otherwise preserve
``input[i]``. Require ``-N < distance < N`` and one physical worker per logical
item. Another overload accepts ``ct::tile<int, ct::shape<1>>`` for ``distance``.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> shuffle_rotate(ct::tile<T, ct::shape<N>> input, int distance = 1, shuffle_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> shuffle_rotate(ct::tile<T, ct::shape<N>> input,
                                            int distance = 1,
                                            shuffle_policy<> = {}) noexcept;

Return ``input[(i + distance) % N]``. Require ``0 <= distance < N`` and one
physical worker per logical item. Another overload accepts ``ct::tile<int,
ct::shape<1>>`` for ``distance``.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> adjacent_difference_subtract_left(ct::tile<T, ct::shape<N>> input, adjacent_difference_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> adjacent_difference_subtract_left(
       ct::tile<T, ct::shape<N>> input,
       adjacent_difference_policy<> = {}) noexcept;

Return the current value minus its predecessor; preserve the first input value.
Shape and dtype are unchanged. No custom operation or external predecessor is
accepted.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> adjacent_difference_subtract_right(ct::tile<T, ct::shape<N>> input, adjacent_difference_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> adjacent_difference_subtract_right(
       ct::tile<T, ct::shape<N>> input,
       adjacent_difference_policy<> = {}) noexcept;

Return the current value minus its successor; preserve the last input value. Shape
and dtype are unchanged. No custom operation or external successor is accepted.

.. cpp:function:: template <typename T, size_t N> ct::tile<int, ct::shape<N>> discontinuity_flag_heads(ct::tile<T, ct::shape<N>> input, discontinuity_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<int, ct::shape<N>>
   discontinuity_flag_heads(ct::tile<T, ct::shape<N>> input,
                            discontinuity_policy<> = {}) noexcept;

Return an ``int`` flag Tile: one at the first position and where a value differs
from its predecessor; zero elsewhere.

.. cpp:function:: template <typename T, size_t N> ct::tile<int, ct::shape<N>> discontinuity_flag_tails(ct::tile<T, ct::shape<N>> input, discontinuity_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<int, ct::shape<N>>
   discontinuity_flag_tails(ct::tile<T, ct::shape<N>> input,
                            discontinuity_policy<> = {}) noexcept;

Return an ``int`` flag Tile: one at the last position and where a value differs
from its successor; zero elsewhere.

.. cpp:function:: template <typename T, size_t N> discontinuity_pair_result<N> discontinuity_flag_heads_and_tails(ct::tile<T, ct::shape<N>> input, discontinuity_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   discontinuity_pair_result<N>
   discontinuity_flag_heads_and_tails(ct::tile<T, ct::shape<N>> input,
                                      discontinuity_policy<> = {}) noexcept;

Return two ``int`` flag Tiles, heads first and tails second. Use a structured
binding. All run boundaries are local to the input Tile.

See :doc:`/tile/cpp/primitives/exchange_shuffle` and
:doc:`/tile/cpp/primitives/adjacent_discontinuity` for examples and boundaries.

Histogram and decoding
----------------------

.. cpp:function:: template <size_t Bins = 256, typename T, size_t N> ct::tile<int, ct::shape<Bins>> histogram(ct::tile<T, ct::shape<N>> input, histogram_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <size_t Bins = 256, typename T, size_t N>
   ct::tile<int, ct::shape<Bins>> histogram(ct::tile<T, ct::shape<N>> input,
                                            histogram_policy<> = {}) noexcept;

Count unsigned-byte samples into an ``int`` Tile of extent ``Bins``. ``T`` must be
``unsigned char`` and each sample must be less than ``Bins``. ``Bins`` is a
compile-time value of 32, 64, 128, or 256. The selected worker count must divide
both ``N`` and ``Bins``. Counts start at zero for each call.

.. cpp:function:: template <typename T, size_t N> ct::tile<int, ct::shape<256>> histogram_256(ct::tile<T, ct::shape<N>> input, histogram_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<int, ct::shape<256>>
   histogram_256(ct::tile<T, ct::shape<N>> input,
                 histogram_policy<> = {}) noexcept;

Count unsigned-byte samples into all 256 bins. This is the fixed-bin counterpart of
``histogram<256>`` with the same input and result contract.

.. cpp:function:: template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN> ct::tile<ValueT, ct::shape<DecodedN>> run_length_decode( ct::tile<ValueT, ct::shape<ValueRunN>> run_values, ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths, run_length_decode_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <size_t DecodedN, typename ValueT, typename LengthT,
             size_t ValueRunN, size_t LengthRunN>
   ct::tile<ValueT, ct::shape<DecodedN>>
   run_length_decode(ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
                     ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
                     run_length_decode_policy<> = {}) noexcept;

Expand runs into a value Tile of extent ``DecodedN``. Input extents must match.
``LengthT`` is ``unsigned int``, ``unsigned long``, or ``unsigned long long``; run-length sums must fit that type. An overload accepts ``decoded_window_offset``
between ``run_lengths`` and the policy: an integral scalar convertible to
``LengthT`` or ``ct::tile<LengthT, ct::shape<1>>``. The default offset is zero. Use
a nonnegative representable offset. Zero-length runs may only follow all positive
runs. Output beyond the expanded sequence is zero.

.. cpp:function:: template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN> run_length_decode_with_offsets_result<ValueT, LengthT, DecodedN> run_length_decode_with_offsets(ct::tile<ValueT, ct::shape<ValueRunN>> run_values, ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths, run_length_decode_policy<> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <size_t DecodedN, typename ValueT, typename LengthT,
             size_t ValueRunN, size_t LengthRunN>
   run_length_decode_with_offsets_result<ValueT, LengthT, DecodedN>
   run_length_decode_with_offsets(
       ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
       ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
       run_length_decode_policy<> = {}) noexcept;

Return decoded values and relative offsets within their runs, in that order. The
inputs, window-offset overload, and capacity follow ``run_length_decode``. Offsets
have element type ``LengthT``; padded positions contain its maximum value. Both
input and output extents must admit a common worker count.

See :doc:`/tile/cpp/primitives/histogram` and :doc:`/tile/cpp/primitives/run_length_decode`
for complete dtype, extent, and padding contracts.

Ordering and selection
----------------------

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> merge_sort_keys(ct::tile<T, ct::shape<N>> keys, merge_sort_policy<> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>>
   merge_sort_keys(ct::tile<T, ct::shape<N>> keys,
                   merge_sort_policy<> policy = {}) noexcept;

Return keys in ascending numeric order. Overloads optionally accept
``cuda::std::less<CompareT>`` or ``cuda::std::greater<CompareT>`` after ``keys``,
followed optionally by the pair ``valid_items, oob_default``. The control types and
partial-sort contract are described below.

.. cpp:function:: template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN> merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs( ct::tile<KeyT, ct::shape<KeyN>> keys, ct::tile<ValueT, ct::shape<ValueN>> values, merge_sort_policy<> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN>
   merge_sort_pair_result<KeyT, ValueT, KeyN>
   merge_sort_pairs(ct::tile<KeyT, ct::shape<KeyN>> keys,
                    ct::tile<ValueT, ct::shape<ValueN>> values,
                    merge_sort_policy<> policy = {}) noexcept;

Sort keys and associated payloads, returning keys first. The overload forms match
``merge_sort_keys``, with ``values`` immediately after ``keys``. Key and value
extents must match; dtypes may differ. Partial sorting zeroes padded payloads.
Equal keys have no stability guarantee.

.. cpp:function:: template <typename T, size_t N, size_t RadixBits = 4, bool MemoizeOuterScan = true, scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans, radix_smem_config SmemConfig = radix_smem_config::four_byte> ct::tile<T, ct::shape<N>> radix_sort_keys(ct::tile<T, ct::shape<N>> keys, radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N, size_t RadixBits = 4,
             bool MemoizeOuterScan = true,
             scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
             radix_smem_config SmemConfig = radix_smem_config::four_byte>
   ct::tile<T, ct::shape<N>>
   radix_sort_keys(ct::tile<T, ct::shape<N>> keys,
                   radix_sort_policy<RadixBits, MemoizeOuterScan,
                                     InnerScanAlgorithm, SmemConfig>
                       policy = {}) noexcept;

Sort keys in ascending radix order over the full key width. Overloads optionally
accept ``cuda::std::less<CompareT>`` or ``cuda::std::greater<CompareT>`` after
``keys``, followed optionally by ``begin_bit, end_bit``. The bounds and comparison
may each be omitted independently. Bounds accept the scalar/Tile forms described
below.

.. cpp:function:: template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN, size_t RadixBits = 4, bool MemoizeOuterScan = true, scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans, radix_smem_config SmemConfig = radix_smem_config::four_byte> radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs( ct::tile<KeyT, ct::shape<KeyN>> keys, ct::tile<ValueT, ct::shape<ValueN>> values, radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN,
             size_t RadixBits = 4, bool MemoizeOuterScan = true,
             scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
             radix_smem_config SmemConfig = radix_smem_config::four_byte>
   radix_sort_pair_result<KeyT, ValueT, KeyN>
   radix_sort_pairs(ct::tile<KeyT, ct::shape<KeyN>> keys,
                    ct::tile<ValueT, ct::shape<ValueN>> values,
                    radix_sort_policy<RadixBits, MemoizeOuterScan,
                                      InnerScanAlgorithm, SmemConfig>
                        policy = {}) noexcept;

Sort keys and associated values, preserving equal-digit input order. Its overload
forms match ``radix_sort_keys``, with ``values`` immediately after ``keys``. Key
and value extents must match; dtypes may differ.

.. cpp:function:: template <typename T, size_t N, size_t RadixBits = 4, bool Descending = false, bool MemoizeOuterScan = true, scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans, radix_smem_config SmemConfig = radix_smem_config::four_byte> ct::tile<int, ct::shape<N>> radix_rank_keys(ct::tile<T, ct::shape<N>> keys, radix_rank_policy<RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N, size_t RadixBits = 4,
             bool Descending = false, bool MemoizeOuterScan = true,
             scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
             radix_smem_config SmemConfig = radix_smem_config::four_byte>
   ct::tile<int, ct::shape<N>>
   radix_rank_keys(ct::tile<T, ct::shape<N>> keys,
              radix_rank_policy<RadixBits, Descending, MemoizeOuterScan,
                                InnerScanAlgorithm, SmemConfig> = {}) noexcept;

Return an ``int`` destination rank per original input item for one stable radix
digit. Keys must be integral. Overloads accept no bounds, ``begin_bit`` alone, or
``begin_bit, end_bit`` before the policy. Scalar controls use ``int``; one or both
bounds may instead be shape-one ``int`` Tiles. The omitted end is ``min(key_bits,
begin_bit + RadixBits)``. Signed keys have their sign bit flipped before
extraction, matching CUB's radix sort ordering.
Unsigned keys use their original bits.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> topk_max_keys( ct::tile<T, ct::shape<N>> keys, int k, topk_policy<> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> topk_max_keys(ct::tile<T, ct::shape<N>> keys,
                                           int k,
                                           topk_policy<> policy = {}) noexcept;

Select up to ``k`` of the largest keys. The output retains input capacity;
only its first ``min(k, valid_items)`` positions are defined and their order
is unspecified. The four numerical control forms below are available before
the final optional policy.

.. cpp:function:: template <typename T, size_t N> ct::tile<T, ct::shape<N>> topk_min_keys( ct::tile<T, ct::shape<N>> keys, int k, topk_policy<> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename T, size_t N>
   ct::tile<T, ct::shape<N>> topk_min_keys(ct::tile<T, ct::shape<N>> keys,
                                           int k,
                                           topk_policy<> policy = {}) noexcept;

Select up to ``k`` of the smallest keys. The output retains input capacity;
only its first ``min(k, valid_items)`` positions are defined and their order
is unspecified. The four numerical control forms below are available before
the final optional policy.

.. cpp:function:: template <typename KeyT, typename ValueT, size_t N> topk_pair_result<KeyT, ValueT, N> topk_max_pairs( ct::tile<KeyT, ct::shape<N>> keys, ct::tile<ValueT, ct::shape<N>> values, int k, topk_policy<> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t N>
   topk_pair_result<KeyT, ValueT, N>
   topk_max_pairs(ct::tile<KeyT, ct::shape<N>> keys,
                  ct::tile<ValueT, ct::shape<N>> values, int k,
                  topk_policy<> policy = {}) noexcept;

Select the largest ``k`` keys with associated payloads. The output retains input
capacity; only its first ``min(k, valid_items)`` positions are defined and their
order is unspecified. The four numerical-control forms below are available before the final
optional policy. Pair results return keys first; input shapes must match.

.. cpp:function:: template <typename KeyT, typename ValueT, size_t N> topk_pair_result<KeyT, ValueT, N> topk_min_pairs( ct::tile<KeyT, ct::shape<N>> keys, ct::tile<ValueT, ct::shape<N>> values, int k, topk_policy<> policy = {}) noexcept
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t N>
   topk_pair_result<KeyT, ValueT, N>
   topk_min_pairs(ct::tile<KeyT, ct::shape<N>> keys,
                  ct::tile<ValueT, ct::shape<N>> values, int k,
                  topk_policy<> policy = {}) noexcept;

Select the smallest ``k`` keys with associated payloads. The output retains input
capacity; only its first ``min(k, valid_items)`` positions are defined and their
order is unspecified. The four numerical-control forms below are available before the final
optional policy. Pair results return keys first; input shapes must match.

Sorting controls
~~~~~~~~~~~~~~~~

For comparison sorting, ``CompareT`` is ``void`` or exactly the key type.
The valid prefix must satisfy ``0 <= valid_items <= N``. Supply both
``valid_items`` and ``oob_default`` together. The scalar forms are ``int``
and a value convertible to the key dtype. Either or both may instead be
shape-one Tiles: ``ct::tile<int, ct::shape<1>>`` for the count and a Tile
of exactly the key dtype for the padding key. Choose a strict high sentinel
for ascending sort or a strict low sentinel for descending sort when
consuming the valid prefix. See :doc:`/tile/cpp/primitives/merge_sort`.

Radix sort requires ``N <= 65535`` and
``0 <= begin_bit <= end_bit <= key_bits``; an empty interval compares
all keys equally. Radix rank has the same extent bound and requires
``0 <= begin_bit < end_bit <= key_bits`` and
``end_bit - begin_bit <= RadixBits``. Sort and rank bit controls are
``int`` scalars or shape-one ``int`` Tiles, including mixed scalar/Tile
bounds. Policies follow all controls. See :doc:`/tile/cpp/primitives/radix_sort_rank`
for signed and floating-point ordering and the distinction between rank
and sort.

TopK overloads
~~~~~~~~~~~~~~

Each of the four TopK functions accepts the following argument sequences
after its data Tiles. These are call forms, not additional declarations:

.. code-block:: text

   k, policy = {}
   k, valid_items, policy = {}
   k, begin_bit, end_bit, policy = {}
   k, valid_items, begin_bit, end_bit, policy = {}

All four numerical controls are ``int`` scalars; shape-one Tile controls
are unsupported. Require ``0 <= k <= N`` and ``0 <= valid_items <= N`` and
``0 <= begin_bit < end_bit <= key_bits``. Omitting ``valid_items`` uses
``N``; omitting bit bounds selects the full key. Tied payload identity is
unspecified. See :doc:`/tile/cpp/primitives/topk` for selected-prefix consumption.

Pair result types
-----------------

These aliases use ``cte::multi_outputs`` and support structured bindings.
They describe returned Tile values; they do not allocate output arrays.

.. cpp:type:: template<typename KeyT, typename ValueT, size_t N> topk_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>, ct::tile<ValueT, ct::shape<N>>>
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t N>
   using topk_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>,
                                               ct::tile<ValueT, ct::shape<N>>>;

Selected keys, then their payloads. Only the selected prefix is defined.

.. cpp:type:: template<typename KeyT, typename ValueT, size_t N> merge_sort_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>, ct::tile<ValueT, ct::shape<N>>>
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t N>
   using merge_sort_pair_result =
       cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>,
                          ct::tile<ValueT, ct::shape<N>>>;

Sorted keys, then their associated payloads.

.. cpp:type:: template<typename KeyT, typename ValueT, size_t N> radix_sort_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>, ct::tile<ValueT, ct::shape<N>>>
   :no-typesetting:

.. code-block:: cpp

   template <typename KeyT, typename ValueT, size_t N>
   using radix_sort_pair_result =
       cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>,
                          ct::tile<ValueT, ct::shape<N>>>;

Radix-sorted keys, then their associated payloads.

.. cpp:type:: template<size_t N> discontinuity_pair_result = cte::multi_outputs<ct::tile<int, ct::shape<N>>, ct::tile<int, ct::shape<N>>>
   :no-typesetting:

.. code-block:: cpp

   template <size_t N>
   using discontinuity_pair_result =
       cte::multi_outputs<ct::tile<int, ct::shape<N>>,
                          ct::tile<int, ct::shape<N>>>;

Head flags, then tail flags.

.. cpp:type:: template<typename ValueT, typename LengthT, size_t N> run_length_decode_with_offsets_result = cte::multi_outputs<ct::tile<ValueT, ct::shape<N>>, ct::tile<LengthT, ct::shape<N>>>
   :no-typesetting:

.. code-block:: cpp

   template <typename ValueT, typename LengthT, size_t N>
   using run_length_decode_with_offsets_result =
       cte::multi_outputs<ct::tile<ValueT, ct::shape<N>>,
                          ct::tile<LengthT, ct::shape<N>>>;

Decoded values, then relative offsets within their runs.

Operation guides
----------------

.. toctree::
   :maxdepth: 1

   primitives/scan
   primitives/reduce
   primitives/exchange_shuffle
   primitives/adjacent_discontinuity
   primitives/histogram
   primitives/run_length_decode
   primitives/merge_sort
   primitives/radix_sort_rank
   primitives/topk

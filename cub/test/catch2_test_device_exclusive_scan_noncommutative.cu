// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <thrust/tabulate.h>

#include <c2h/catch2_test_helper.h>
#include <catch2_test_device_scan.cuh>

/* Consider free monoid with two generators, ``q`` and ``p``, modulo defining relationship (``p * q == 1``).
 * Elements of this algebra are ``q^m * p^n``, identified by a pair of integral exponents. The identity
 * element is ``1 == q^0 * p^0``, which maps to pair of zeros ``e = (0, 0)``.
 *
 * The product is defined by concatenation:
 *     q^m * p^n * q^r * p^s == q^m * p^{n-1} * p * q * q^{r-1} * p^s
 *                           == q^m * p^{n-1} * q^{r-1} * p^s
 *
 * This reduction can be performed ``min(n, r)`` times resulting in
 *
 *     q^m * p^n * q^r * p^s == q^{m + r - min(n, r)} * p^{s + n - min(n, r)}
 *
 * Hence this is a monoid, known as bicyclic monoid.
 * This operation on pairs of integers is associative (since concatenation is), but non-commutative.
 *
 *    Ref: https://en.wikipedia.org/wiki/Bicyclic_semigroup
 *    Ref: https://en.wikipedia.org/wiki/Monoid
 */

/**
 * PropagateLastWrite: an associative, non-commutative operator over a symbol alphabet.
 *
 * Given a read_symbol r, the operator is:
 *     op(lhs, rhs) = lhs  if  rhs == r  and  lhs != r
 *     op(lhs, rhs) = rhs  otherwise
 *
 * Semantics: the output carries the most recent "write" symbol seen so far.
 * Reads (rhs == r) are replaced by the preceding write (lhs) if one exists.
 *
 * Identity: r itself, because op(r, x) = x for all x and op(x, r) propagates x.
 *
 * Associativity holds because the operator tracks the "last non-read" value
 * seen from the left:
 *   op(a, op(b, c)) == op(op(a, b), c)
 *
 * Non-commutativity: op({write}, {read}) == {write}, but
 *                    op({read}, {write}) == {write} as well —
 * however op({write_A}, {write_B}) == {write_B} while
 *         op({write_B}, {write_A}) == {write_A}, which differ when A != B.
 *
 * This operator is used in applications such as stack-symbol propagation
 * during JSON or bracket parsing. A regression was found in the warpspeed
 * ExclusiveScan kernel (sm_100+) introduced in CCCL 3.4, where the
 * inter-tile prefix was combined with the intra-tile prefix in the wrong
 * order, producing incorrect results for non-commutative operators.
 */

namespace impl
{
// bicyclic monoid operator: associative and non-commutative
template <typename UnsignedIntegralT>
struct bicyclic_monoid_op
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;
  using min_t  = cuda::minimum<>;

  __host__ __device__ pair_t operator()(pair_t v1, pair_t v2) const
  {
    auto [m, n] = v1;
    auto [r, s] = v2;
    auto min_nr = min_t{}(n, r);
    return {m + r - min_nr, s + n - min_nr};
  }
};

// Functor to fill input with a repeating short sequence of bicyclic pairs
template <typename UnsignedIntegralT>
struct populate_bicyclic_input
{
  using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;

  __host__ __device__ pair_t operator()(size_t idx) const
  {
    static constexpr pair_t seq[] = {
      {0, 1}, {2, 3}, {4, 1}, {2, 5}, {7, 1}, {1, 1}, {0, 4}, {3, 1}, {1, 2}, {3, 2}, {4, 5}, {3, 5},
      {1, 9}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {2, 2}, {2, 2}, {0, 0}, {1, 1}, {2, 3},
      {2, 4}, {4, 3}, {1, 3}, {0, 3}, {1, 1}, {5, 1}, {2, 3}, {4, 7}, {2, 6}, {8, 3}, {1, 0}, {0, 8}};

    static constexpr size_t nelems = sizeof(seq) / sizeof(pair_t);
    return seq[idx % nelems];
  }
};

// PropagateLastWrite: associative and non-commutative
template <typename StackSymbolT>
struct propagate_last_write_op
{
  StackSymbolT read_symbol;

  __host__ __device__ StackSymbolT operator()(StackSymbolT lhs, StackSymbolT rhs) const
  {
    return (rhs == read_symbol && lhs != read_symbol) ? lhs : rhs;
  }
};

// Functor to fill input: writes every write_period positions, reads everywhere else
template <typename StackSymbolT>
struct populate_sparse_write_input
{
  StackSymbolT read_symbol;
  StackSymbolT write_symbol;
  int write_period;

  __host__ __device__ StackSymbolT operator()(size_t idx) const
  {
    return (idx % static_cast<size_t>(write_period) == 0) ? write_symbol : read_symbol;
  }
};
} // namespace impl

// Sizes chosen to cover: single-element, small, medium, the original bug-report size (8160),
// and large enough to span multiple warpspeed tiles on any supported architecture.
// On sm_100+ the warpspeed tile size is num_scan_stor_threads * items_per_thread, which is
// ~4000–32000 elements depending on element size, so 50'000 guarantees multiple tiles for all types.

C2H_TEST("Device exclusive scan works with non-commutative bicyclic monoid operator", "[scan][device]")
{
  using pair_t = cuda::std::pair<unsigned, unsigned>;
  using op_t   = impl::bicyclic_monoid_op<unsigned>;

  const int num_items = GENERATE_COPY(1, 10, 1337, 3000, 8160, 50'000, 1'000'000);

  CAPTURE(num_items);

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_bicyclic_input<unsigned>{});
  c2h::device_vector<pair_t> output(num_items);

  pair_t* d_input  = thrust::raw_pointer_cast(input.data());
  pair_t* d_output = thrust::raw_pointer_cast(output.data());

  const pair_t init_value{0, 0}; // identity element of the bicyclic monoid

  size_t tmp_size{};
  cudaError_t status1 =
    cub::DeviceScan::ExclusiveScan(nullptr, tmp_size, d_input, d_output, op_t{}, init_value, num_items);
  REQUIRE(cudaSuccess == status1);
  REQUIRE(tmp_size > 0);

  using cuda::std::byte;

  c2h::device_vector<byte> tmp(tmp_size);
  byte* d_tmp = thrust::raw_pointer_cast(tmp.data());

  cudaError_t status2 =
    cub::DeviceScan::ExclusiveScan(d_tmp, tmp_size, d_input, d_output, op_t{}, init_value, num_items);
  REQUIRE(cudaSuccess == status2);

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(num_items);

  compute_exclusive_scan_reference(h_input.cbegin(), h_input.cend(), h_expected.begin(), init_value, op_t{});

  REQUIRE(h_expected == h_output);
}

C2H_TEST("Device exclusive scan works with PropagateLastWrite operator", "[scan][device]")
{
  // PropagateLastWrite<char> uses char values: contiguous, trivially copyable, arithmetic —
  // all conditions that enable the warpspeed kernel on sm_100+.
  // The bug (CCCL 3.4 regression) was that the warpspeed kernel applied scan_op with operands
  // in the wrong order when combining the inter-tile exclusive prefix with the intra-tile prefix,
  // producing incorrect results for non-commutative operators like this one.

  using symbol_t                        = char;
  constexpr symbol_t read_symbol        = 'x';
  constexpr symbol_t empty_stack_symbol = '_'; // init_value for the exclusive scan
  constexpr symbol_t write_symbol       = '[';

  using op_t = impl::propagate_last_write_op<symbol_t>;
  const op_t op{read_symbol};

  // A write period of 97 (prime) ensures writes do not align with tile or warp boundaries,
  // stressing the cross-tile prefix propagation through any power-of-two tiling scheme.
  constexpr int write_period = 97;

  const int num_items = GENERATE_COPY(1, 10, 1337, 3000, 8160, 50'000, 1'000'000);

  CAPTURE(num_items);

  c2h::device_vector<symbol_t> input(num_items);
  thrust::tabulate(
    input.begin(), input.end(), impl::populate_sparse_write_input<symbol_t>{read_symbol, write_symbol, write_period});
  c2h::device_vector<symbol_t> output(num_items);

  symbol_t* d_input  = thrust::raw_pointer_cast(input.data());
  symbol_t* d_output = thrust::raw_pointer_cast(output.data());

  size_t tmp_size{};
  cudaError_t status1 =
    cub::DeviceScan::ExclusiveScan(nullptr, tmp_size, d_input, d_output, op, empty_stack_symbol, num_items);
  REQUIRE(cudaSuccess == status1);
  REQUIRE(tmp_size > 0);

  using cuda::std::byte;

  c2h::device_vector<byte> tmp(tmp_size);
  byte* d_tmp = thrust::raw_pointer_cast(tmp.data());

  cudaError_t status2 =
    cub::DeviceScan::ExclusiveScan(d_tmp, tmp_size, d_input, d_output, op, empty_stack_symbol, num_items);
  REQUIRE(cudaSuccess == status2);

  // transfer to host_vector is synchronizing
  c2h::host_vector<symbol_t> h_output(output);
  c2h::host_vector<symbol_t> h_input(input);
  c2h::host_vector<symbol_t> h_expected(num_items);

  compute_exclusive_scan_reference(h_input.cbegin(), h_input.cend(), h_expected.begin(), empty_stack_symbol, op);

  REQUIRE(h_expected == h_output);
}

C2H_TEST("Device exclusive scan PropagateLastWrite reproduces original bug-report input", "[scan][device]")
{
  // Regression test using the exact input pattern from the original bug report.
  // This input has a cluster of write symbols near the end that must be correctly
  // propagated across a tile boundary by the inter-tile lookback mechanism.

  using symbol_t                        = char;
  constexpr symbol_t read_symbol        = 'x';
  constexpr symbol_t empty_stack_symbol = '_';
  constexpr int num_elements            = 8160;

  using op_t = impl::propagate_last_write_op<symbol_t>;
  const op_t op{read_symbol};

  c2h::host_vector<symbol_t> h_input(num_elements, read_symbol);

  h_input[8020] = '_';
  h_input[8023] = '{';
  h_input[8057] = '[';
  h_input[8060] = '{';
  h_input[8061] = '[';
  h_input[8068] = '{';
  h_input[8073] = '[';
  h_input[8074] = '{';
  h_input[8075] = '[';
  h_input[8076] = '{';
  h_input[8148] = '_';
  h_input[8151] = '{';
  h_input[8152] = '_';
  h_input[8154] = '[';
  h_input[8155] = '_';
  h_input[8157] = '[';
  h_input[8159] = '_';

  c2h::host_vector<symbol_t> h_expected(num_elements);
  compute_exclusive_scan_reference(h_input.cbegin(), h_input.cend(), h_expected.begin(), empty_stack_symbol, op);

  c2h::device_vector<symbol_t> input(h_input);
  c2h::device_vector<symbol_t> output(num_elements);

  symbol_t* d_input  = thrust::raw_pointer_cast(input.data());
  symbol_t* d_output = thrust::raw_pointer_cast(output.data());

  size_t tmp_size{};
  cudaError_t status1 =
    cub::DeviceScan::ExclusiveScan(nullptr, tmp_size, d_input, d_output, op, empty_stack_symbol, num_elements);
  REQUIRE(cudaSuccess == status1);

  using cuda::std::byte;

  c2h::device_vector<byte> tmp(tmp_size);
  byte* d_tmp = thrust::raw_pointer_cast(tmp.data());

  cudaError_t status2 =
    cub::DeviceScan::ExclusiveScan(d_tmp, tmp_size, d_input, d_output, op, empty_stack_symbol, num_elements);
  REQUIRE(cudaSuccess == status2);

  c2h::host_vector<symbol_t> h_output(output);
  REQUIRE(h_expected == h_output);
}

// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_scan.cuh>

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
 * This operation of pairs of integers is associative (since concatenation is), but non-commutative.
 *
 *    Ref: https://en.wikipedia.org/wiki/Bicyclic_semigroup
 *    Ref: https://en.wikipedia.org/wiki/Monoid
 */

namespace impl
{
// bicyclid monoid operator is associative and non-commutative
template <typename UnsignedIntegralT>
struct bicyclic_monoid_op
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;
  using min_t  = cuda::minimum<>;

  pair_t __host__ __device__ operator()(pair_t v1, pair_t v2)
  {
    auto [m, n] = v1;
    auto [r, s] = v2;
    auto min_nr = min_t{}(n, r);
    return {m + r - min_nr, s + n - min_nr};
  }
};

template <typename UnsignedIntegralT>
struct populate_input
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;

  __host__ __device__ pair_t operator()(size_t id) const
  {
    static constexpr pair_t short_seq[] = {
      {0, 1}, {2, 3}, {4, 1}, {2, 5}, {7, 1}, {1, 1}, {0, 4}, {3, 1}, {1, 2}, {3, 2}, {4, 5}, {3, 5},
      {1, 9}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {2, 2}, {2, 2}, {0, 0}, {1, 1}, {2, 3},
      {2, 4}, {4, 3}, {1, 3}, {0, 3}, {1, 1}, {5, 1}, {2, 3}, {4, 7}, {2, 6}, {8, 3}, {1, 0}, {0, 8}};

    static constexpr size_t nelems = sizeof(short_seq) / sizeof(pair_t);

    return short_seq[id % nelems];
  }
};
}; // namespace impl

C2H_TEST("Device inclusive segmented scan works with non-commutative operator", "[segmented][scan][device]")
{
  using pair_t = cuda::std::pair<unsigned, unsigned>;
  using op_t   = impl::bicyclic_monoid_op<unsigned>;

  unsigned num_items = 1'234'567;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  size_t tmp_size{};
  cudaError_t status1 = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    nullptr, tmp_size, d_input, d_output, d_offsets, d_offsets + 1, num_segments, op_t{});
  REQUIRE(cudaSuccess == status1);
  REQUIRE(tmp_size > 0);

  using cuda::std::byte;

  c2h::device_vector<byte> tmp(tmp_size, thrust::no_init);
  byte* d_tmp = thrust::raw_pointer_cast(tmp.data());

  REQUIRE(d_tmp != nullptr);

  cudaError_t status2 = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    d_tmp, tmp_size, d_input, d_output, d_offsets, d_offsets + 1, num_segments, op_t{});
  REQUIRE(cudaSuccess == status2);

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op_t{},
      pair_t{0, 0});
  }

  REQUIRE(h_expected == h_output);
}

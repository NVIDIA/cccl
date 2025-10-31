// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
struct BicyclicMonoidOp
{
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

}; // namespace impl

C2H_TEST("Device inclusive scan works with non-commutative operator", "[scan][device]")
{
  using pair_t = cuda::std::pair<unsigned, unsigned>;
  using op_t   = impl::BicyclicMonoidOp<unsigned>;

  thrust::device_vector<pair_t> input{
    {0, 1}, {2, 3}, {4, 1}, {2, 5}, {7, 1}, {1, 1}, {0, 4}, {3, 1}, {1, 2}, {3, 2}, {4, 5}, {3, 5},
    {1, 9}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {2, 2}, {2, 2}, {0, 0}, {1, 1}, {2, 3}};
  thrust::device_vector<pair_t> output(input.size());

  pair_t* d_input  = thrust::raw_pointer_cast(input.data());
  pair_t* d_output = thrust::raw_pointer_cast(output.data());

  size_t tmp_size{};
  cudaError_t status1 = cub::DeviceScan::InclusiveScan(nullptr, tmp_size, d_input, d_output, op_t{}, input.size());
  REQUIRE(cudaSuccess == status1);
  REQUIRE(tmp_size > 0);

  using _byte_t = cuda::std::uint8_t;

  thrust::device_vector<_byte_t> tmp(tmp_size);
  _byte_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  REQUIRE(d_tmp != nullptr);

  cudaError_t status2 = cub::DeviceScan::InclusiveScan(d_tmp, tmp_size, d_input, d_output, op_t{}, input.size());
  REQUIRE(cudaSuccess == status2);

  // transfer to host_vector is synchronizing
  thrust::host_vector<pair_t> h_output(output);
  thrust::host_vector<pair_t> h_input(input);
  thrust::host_vector<pair_t> h_expected(input.size());

  compute_inclusive_scan_reference(h_input.begin(), h_input.end(), h_expected.begin(), op_t{}, pair_t{0, 0});

  REQUIRE(h_expected == h_output);
}

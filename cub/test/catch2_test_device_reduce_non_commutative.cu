// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/functional>

#include <cstdint>
#include <numeric>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_reduce_non_commutative.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceNonCommutative, reduce_non_commutative);

// %PARAM% TEST_LAUNCH lid 0:1:2

// The affine map x -> a * x + b. Composing maps is associative but not commutative, and with an odd `a` every map is
// invertible, so a different order of composition almost always gives a different map.
template <typename T>
struct affine_t
{
  T a;
  T b;

  __host__ __device__ friend bool operator==(const affine_t& lhs, const affine_t& rhs)
  {
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }

  friend std::ostream& operator<<(std::ostream& os, const affine_t& f)
  {
    return os << '{' << +f.a << ", " << +f.b << '}';
  }
};

// Applies `lhs` first, then `rhs`
template <typename T>
struct compose_t
{
  __host__ __device__ affine_t<T> operator()(const affine_t<T>& lhs, const affine_t<T>& rhs) const
  {
    // Promote small types to unsigned int, so the products wrap around instead of overflowing int
    using wide_t = decltype(T{} * 1u);
    return {static_cast<T>(wide_t{rhs.a} * lhs.a), static_cast<T>(wide_t{rhs.a} * lhs.b + rhs.b)};
  }
};

template <typename T>
struct random_to_affine_t
{
  __host__ __device__ affine_t<T> operator()(std::uint64_t value) const
  {
    return {static_cast<T>(value | 1), static_cast<T>(value >> 32)};
  }
};

using affine_types =
  c2h::type_list<affine_t<std::uint8_t>, affine_t<std::uint16_t>, affine_t<std::uint32_t>, affine_t<std::uint64_t>>;

CUB_TEST("Device reduce non-commutative combines items in input order", "[reduce][device]", CUB_SMALL)
{
  const auto num_items = GENERATE_COPY(values<std::int64_t>({0, 1, 2, 31, 32, 33, 4095, 4096, 4097, 65536, 65537}),
                                       take(3, random(std::int64_t{1} << 17, std::int64_t{1} << 24)));
  CAPTURE(num_items);

  const auto d_in = cuda::make_transform_iterator(cuda::counting_iterator<std::int64_t>{0}, index_to_run_t{});
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  reduce_non_commutative(d_in, d_out.begin(), num_items, concatenate_runs_t{}, initial_run);

  REQUIRE(d_out[0] == expected_run(num_items));
}

CUB_TEST("Device reduce non-commutative works with a pointer to the items", "[reduce][device]", CUB_SMALL)
{
  const auto num_items = GENERATE_COPY(
    values<std::int64_t>({1, 2, 33, 4097}), take(3, random(std::int64_t{1} << 12, std::int64_t{1} << 22)));
  const auto offset = GENERATE(0, 1); // an offset of one item makes the input misaligned for vector loads
  CAPTURE(num_items, offset);

  c2h::device_vector<run_t> d_in(num_items + offset, thrust::no_init);
  thrust::transform(
    c2h::device_policy,
    cuda::counting_iterator<std::int64_t>{-offset},
    cuda::counting_iterator<std::int64_t>{num_items},
    d_in.begin(),
    index_to_run_t{});
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  reduce_non_commutative(
    thrust::raw_pointer_cast(d_in.data()) + offset, d_out.begin(), num_items, concatenate_runs_t{}, initial_run);

  REQUIRE(d_out[0] == expected_run(num_items));
}

CUB_TEST("Device reduce non-commutative composes affine maps in order", "[reduce][device]", CUB_SMALL, affine_types)
{
  using item_t = c2h::get<0, TestType>;
  using op_t   = compose_t<decltype(item_t{}.a)>;

  const int num_items =
    GENERATE_COPY(values({1, 2, 3, 31, 32, 33, 4095, 4096, 4097}), take(3, random(1 << 12, 1 << 22)));
  const int offset = GENERATE(0, 1);
  CAPTURE(c2h::type_name<item_t>(), num_items, offset);

  c2h::device_vector<std::uint64_t> random_values(num_items + offset, thrust::no_init);
  c2h::gen(C2H_SEED(2), random_values);
  c2h::device_vector<item_t> d_in(num_items + offset, thrust::no_init);
  thrust::transform(
    c2h::device_policy,
    random_values.begin(),
    random_values.end(),
    d_in.begin(),
    random_to_affine_t<decltype(item_t{}.a)>{});
  c2h::device_vector<item_t> d_out(1, thrust::no_init);

  const item_t identity{1, 0};
  reduce_non_commutative(thrust::raw_pointer_cast(d_in.data()) + offset, d_out.begin(), num_items, op_t{}, identity);

  const c2h::host_vector<item_t> h_in(d_in);
  const item_t expected = std::accumulate(h_in.begin() + offset, h_in.end(), identity, op_t{});
  REQUIRE(d_out[0] == expected);
}

CUB_TEST("Device reduce non-commutative applies the initial value first", "[reduce][device]", CUB_SMALL)
{
  using item_t = affine_t<std::uint32_t>;

  const int num_items = GENERATE(1, 4097, 1 << 20);
  CAPTURE(num_items);

  c2h::device_vector<std::uint64_t> random_values(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), random_values);
  c2h::device_vector<item_t> d_in(num_items, thrust::no_init);
  thrust::transform(
    c2h::device_policy, random_values.begin(), random_values.end(), d_in.begin(), random_to_affine_t<std::uint32_t>{});
  c2h::device_vector<item_t> d_out(1, thrust::no_init);

  const item_t init{3, 5};
  reduce_non_commutative(d_in.begin(), d_out.begin(), num_items, compose_t<std::uint32_t>{}, init);

  const c2h::host_vector<item_t> h_in(d_in);
  REQUIRE(d_out[0] == std::accumulate(h_in.begin(), h_in.end(), init, compose_t<std::uint32_t>{}));
}

CUB_TEST("Device reduce non-commutative works with commutative operators", "[reduce][device]", CUB_SMALL)
{
  const int num_items = GENERATE_COPY(values({1, 4097}), take(2, random(1 << 12, 1 << 22)));
  CAPTURE(num_items);

  c2h::device_vector<std::uint8_t> d_in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(2), d_in);
  c2h::device_vector<std::int32_t> d_out(1, thrust::no_init);

  // Adding std::uint8_t items accumulates in int
  reduce_non_commutative(d_in.begin(), d_out.begin(), num_items, cuda::std::plus<>{}, std::uint8_t{0});

  const c2h::host_vector<std::uint8_t> h_in(d_in);
  REQUIRE(d_out[0] == std::accumulate(h_in.begin(), h_in.end(), std::int32_t{0}));
}

using offset_types = c2h::type_list<std::int32_t, std::uint32_t, std::uint64_t>;

CUB_TEST("Device reduce non-commutative works with large offsets", "[reduce][device]", CUB_SMALL, offset_types)
{
  using offset_t = c2h::get<0, TestType>;

  const offset_t num_items = detail::make_large_offset<offset_t>();
  CAPTURE(c2h::type_name<offset_t>(), num_items);

  const auto d_in = cuda::make_transform_iterator(cuda::counting_iterator<std::int64_t>{0}, index_to_run_t{});
  c2h::device_vector<run_t> d_out(1, thrust::no_init);

  reduce_non_commutative(d_in, d_out.begin(), num_items, concatenate_runs_t{}, initial_run);

  REQUIRE(d_out[0] == expected_run(static_cast<std::int64_t>(num_items)));
}

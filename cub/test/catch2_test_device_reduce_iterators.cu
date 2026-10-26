// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/sequence.h>

#include <cuda/iterator>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of types to test
using custom_t           = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;
using iterator_type_list = c2h::type_list<type_pair<custom_t>, type_pair<std::int64_t>>;

/**
 * @brief Helper function to test large problem sizes, including problems requiring 64-bit offset
 * types.
 */
template <typename T, typename OffsetT>
void test_big_indices_helper(OffsetT num_items)
{
  const cuda::constant_iterator<T> const_iter(T{1});
  c2h::device_vector<std::size_t> out(1);
  std::size_t* d_out = thrust::raw_pointer_cast(out.data());
  device_sum(const_iter, d_out, num_items);
  const std::size_t result = out[0];

  REQUIRE(result == num_items);
}

CUB_TEST("Device sum works for big indices", "[reduce][device]", CUB_SMALL)
{
  test_big_indices_helper<std::size_t, std::uint32_t>(1ull << 30);
  test_big_indices_helper<std::size_t, std::uint32_t>(1ull << 31);
  test_big_indices_helper<std::size_t, std::uint32_t>((1ull << 32) - 1);
  test_big_indices_helper<std::size_t, std::uint64_t>(1ull << 33);
}

CUB_TEST("Device reduce works with fancy input iterators", "[reduce][device]", CUB_SMALL, iterator_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr int max_items    = 5000000;
  constexpr int min_items    = 1;
  constexpr int num_segments = 1;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Prepare input data
  item_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = cuda::constant_iterator(default_constant);

  using op_t         = cuda::std::plus<>;
  using init_value_t = output_t;

  // Binary reduction operator
  auto reduction_op = op_t{};

  // Prepare verification data
  using accum_t                  = cuda::std::__accumulator_t<op_t, item_t, init_value_t>;
  const output_t expected_result = compute_single_problem_reference(in_it, in_it + num_items, reduction_op, accum_t{});

  // Run test
  c2h::device_vector<output_t> out_result(num_segments);
  auto d_out_it = thrust::raw_pointer_cast(out_result.data());
  device_reduce(in_it, d_out_it, num_items, reduction_op, init_value_t{});

  // Verify result
  REQUIRE(expected_result == out_result[0]);
}

CUB_TEST("Device reduce compiles with discard output iterator", "[reduce][device]", CUB_SMALL, iterator_type_list)
{
  using params   = params_t<TestType>;
  using item_t   = typename params::item_t;
  using output_t = typename params::output_t;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(values({
    min_items,
    max_items,
  }));

  // Prepare input data
  item_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = cuda::constant_iterator(default_constant);

  using op_t         = cuda::std::plus<>;
  using init_value_t = output_t;

  // Binary reduction operator
  auto reduction_op = op_t{};

  // Run test
  device_reduce(in_it, cuda::discard_iterator(), num_items, reduction_op, init_value_t{});
}

struct plus_one_t
{
  __host__ __device__ int operator()(int value) const
  {
    return value + 1;
  }
};

struct widen_t
{
  __host__ __device__ int operator()(std::uint8_t value) const
  {
    return static_cast<int>(value) + 1;
  }
};

template <typename Iterator>
void check_sum(Iterator in, int num_items, std::int64_t expected)
{
  c2h::device_vector<std::int64_t> out(1, thrust::no_init);
  device_sum(in, thrust::raw_pointer_cast(out.data()), num_items);
  REQUIRE(out[0] == expected);
}

// Full tiles exercise the vectorized load. The extra elements exercise the scalar tail. An offset of one int is
// aligned for the element type and misaligned for a 16-byte vector load.
CUB_TEST("Device sum of a transform iterator over contiguous storage", "[reduce][device]", CUB_SMALL)
{
  const int num_items = GENERATE(4096, 1000003, 1 << 20);
  c2h::device_vector<int> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());
  auto* d_in = thrust::raw_pointer_cast(in.data());

  const auto expected = static_cast<std::int64_t>(num_items) * (num_items + 1) / 2;
  check_sum(cuda::transform_iterator(d_in, plus_one_t{}), num_items, expected);

  if (num_items > 1)
  {
    const int tail           = num_items - 1;
    const auto tail_expected = expected - 1;
    check_sum(cuda::transform_iterator(d_in + 1, plus_one_t{}), tail, tail_expected);
  }
}

CUB_TEST("Device sum of a widening transform iterator over contiguous storage", "[reduce][device]", CUB_SMALL)
{
  constexpr int num_items = 1 << 20;
  c2h::device_vector<std::uint8_t> in(num_items, 1);
  auto* d_in = thrust::raw_pointer_cast(in.data());
  check_sum(cuda::transform_iterator(d_in, widen_t{}), num_items, static_cast<std::int64_t>(num_items) * 2);
}

CUB_TEST("Device sum of a counting iterator stays scalar", "[reduce][device]", CUB_SMALL)
{
  constexpr int num_items = 1 << 20;
  const auto expected     = static_cast<std::int64_t>(num_items) * (num_items - 1) / 2;
  check_sum(cuda::counting_iterator{0}, num_items, expected);
}

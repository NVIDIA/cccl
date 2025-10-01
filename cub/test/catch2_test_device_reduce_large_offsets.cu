// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>
#include <cub/thread/thread_operators.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <cuda/std/__algorithm_>

#include <cstdint>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of offset types to test
using offset_types = c2h::type_list<std::int32_t, std::uint32_t, std::uint64_t>;

__host__ __device__ _CCCL_FORCEINLINE uint64_t
get_segmented_guassian_sum(const uint64_t num_items, const uint64_t segment_size)
{
  const uint64_t sum_per_full_segment = (segment_size * (segment_size - 1)) / 2;
  const uint64_t full_segments        = num_items / segment_size;
  const uint64_t index_within_segment = num_items % segment_size;

  const uint64_t sum_within_partial_segment = (index_within_segment * (index_within_segment - 1)) / 2;
  const uint64_t sum_over_full_segments     = full_segments * sum_per_full_segment;
  return sum_within_partial_segment + sum_over_full_segments;
}

template <typename ItemT>
struct mod_op
{
  uint64_t segment_size;

  __host__ __device__ _CCCL_FORCEINLINE uint64_t operator()(const uint64_t index) const
  {
    return static_cast<ItemT>(index % segment_size);
  }
};

struct custom_sum_op
{
  template <typename ItemT>
  __host__ __device__ _CCCL_FORCEINLINE ItemT operator()(const ItemT lhs, const ItemT rhs) const
  {
    return lhs + rhs;
  }
};

C2H_TEST("Device reduce works with all device interfaces", "[reduce][device]", offset_types)
{
  using index_t  = uint64_t;
  using offset_t = typename c2h::get<0, TestType>;

  CAPTURE(c2h::type_name<offset_t>());

  const offset_t num_items_max = detail::make_large_offset<offset_t>();
  const offset_t num_items_min = num_items_max > 10000 ? num_items_max - 10000ULL : offset_t{0};

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    values(
      {num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1), static_cast<offset_t>(3)}),
    take(2, random(num_items_min, num_items_max)));

  // Input data
  const auto index_it = thrust::make_counting_iterator(index_t{});

  SECTION("reduce")
  {
    // Use a custom operator to increase test coverage, different from potentially different code paths used for
    // cub::Sum
    using op_t = custom_sum_op;

    // Segment size (generate a series of: 0, 1, 2, ..., <segment_size-1>,  0, 1, 2, ..., <segment_size-1>, 0, 1, ...)
    const auto segment_size = 1000;

    // Initial value of reduction
    const auto init_val = index_t{42};

    // Binary reduction operator
    const auto reduction_op = op_t{};

    // Prepare verification data
    const index_t expected_result = init_val + get_segmented_guassian_sum(num_items, segment_size);

    // Run test
    c2h::device_vector<index_t> out_result(1);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    const auto d_in_it  = thrust::make_transform_iterator(index_it, mod_op<index_t>{segment_size});

    device_reduce(d_in_it, d_out_it, num_items, reduction_op, init_val);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("sum")
  {
    // Segment size (generate a series of: 0, 1, 2, ..., <segment_size-1>,  0, 1, 2, ..., <segment_size-1>, 0, 1, ...)
    const auto segment_size = 1000;

    // Prepare verification data
    const index_t expected_result = get_segmented_guassian_sum(num_items, segment_size);

    // Run test
    c2h::device_vector<index_t> out_result(1);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    const auto d_in_it  = thrust::make_transform_iterator(index_it, mod_op<index_t>{segment_size});

    device_sum(d_in_it, d_out_it, num_items);

    // Verify result
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("min")
  {
    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<index_t> out_result(1);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    const auto d_in_it  = cuda::std::make_reverse_iterator(index_it + num_items + iterator_offset);

    device_min(d_in_it, d_out_it, num_items);

    // Verify result
    const index_t expected_result = iterator_offset;
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("max")
  {
    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<index_t> out_result(1);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    const auto d_in_it  = index_it + iterator_offset;

    device_max(d_in_it, d_out_it, num_items);

    // Verify result
    const index_t expected_result = num_items + iterator_offset - index_t{1};
    REQUIRE(expected_result == out_result[0]);
  }

  SECTION("argmin")
  {
    using result_t = cuda::std::pair<cuda::std::int64_t, index_t>;

    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<result_t> out_result(1);
    auto d_result_ptr   = thrust::raw_pointer_cast(out_result.data());
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;

    const auto d_in_it = cuda::std::make_reverse_iterator(index_it + num_items + iterator_offset);

    device_arg_min(d_in_it, d_extremum_out, d_index_out, num_items);

    // Verify result
    const index_t expected_value = iterator_offset;
    const auto expected_index    = static_cast<cuda::std::int64_t>(num_items - 1);

    // Verify result
    const result_t gpu_result = out_result[0];
    REQUIRE(expected_value == gpu_result.second);
    REQUIRE(expected_index == gpu_result.first);
  }

  SECTION("argmax")
  {
    using result_t = cuda::std::pair<cuda::std::int64_t, index_t>;

    // Run test
    const index_t iterator_offset = 1000;
    c2h::device_vector<result_t> out_result(1);
    auto d_result_ptr   = thrust::raw_pointer_cast(out_result.data());
    auto d_index_out    = &d_result_ptr->first;
    auto d_extremum_out = &d_result_ptr->second;

    const auto d_in_it = index_it + iterator_offset;

    device_arg_max(d_in_it, d_extremum_out, d_index_out, num_items);

    // Verify result
    const index_t expected_value = iterator_offset + num_items - index_t{1};
    const auto expected_index    = static_cast<cuda::std::int64_t>(num_items - 1);

    // Verify result
    const result_t gpu_result = out_result[0];
    REQUIRE(expected_value == gpu_result.second);
    REQUIRE(expected_index == gpu_result.first);
  }
}

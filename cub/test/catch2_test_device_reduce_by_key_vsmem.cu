// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceByKey, device_reduce_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

using large_type_list =
  c2h::type_list<c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<256>::type>,
                 c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<512>::type>>;

C2H_TEST("Device reduce-by-key works with huge keys", "[by_key][reduce][device]", large_type_list)
{
  using key_t    = typename c2h::get<0, TestType>;
  using value_t  = std::uint32_t;
  using output_t = value_t;
  using offset_t = uint32_t;
  using op_t     = cuda::std::plus<>;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 10000)));

  // Range of segment sizes to generate (a segment is a series of consecutive equal keys)
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = thrust::raw_pointer_cast(segment_keys.data());

  // Generate input data
  c2h::device_vector<value_t> in_values(num_items);
  c2h::gen(C2H_SEED(2), in_values);
  auto d_values_it = thrust::raw_pointer_cast(in_values.data());

  // Binary reduction operator
  auto reduction_op = op_t{};

  // Prepare verification data
  using accum_t = cuda::std::__accumulator_t<op_t, value_t, output_t>;
  c2h::host_vector<output_t> expected_result(num_segments);
  compute_segmented_problem_reference(in_values, segment_offsets, reduction_op, accum_t{}, expected_result.begin());
  c2h::host_vector<key_t> expected_keys = compute_unique_keys_reference(segment_keys);

  // Run test
  c2h::device_vector<offset_t> num_unique_keys(1);
  c2h::device_vector<key_t> out_unique_keys(num_segments);
  c2h::device_vector<output_t> out_result(num_segments);
  auto d_out_it      = thrust::raw_pointer_cast(out_result.data());
  auto d_keys_out_it = thrust::raw_pointer_cast(out_unique_keys.data());
  device_reduce_by_key(
    d_keys_it,
    d_keys_out_it,
    unwrap_it(d_values_it),
    unwrap_it(d_out_it),
    thrust::raw_pointer_cast(num_unique_keys.data()),
    reduction_op,
    num_items);

  // Verify result
  REQUIRE(num_segments == num_unique_keys[0]);
  REQUIRE(expected_result == out_result);
  REQUIRE(expected_keys == out_unique_keys);
}

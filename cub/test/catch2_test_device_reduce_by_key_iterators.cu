// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <cuda/iterator>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceByKey, device_reduce_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of types to test
using custom_t           = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;
using iterator_type_list = c2h::type_list<type_triple<custom_t>, type_triple<std::int64_t, std::int64_t, custom_t>>;

C2H_TEST("Device reduce-by-key works with iterators", "[by_key][reduce][device]", iterator_type_list)
{
  using params   = params_t<TestType>;
  using value_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using key_t    = typename params::type_pair_t::key_t;
  using offset_t = uint32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Number of items
  const offset_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  INFO("Test num_items: " << num_items);

  // Range of segment sizes to generate (a segment is a series of consecutive equal keys)
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));

  // Get array of keys from segment offsets
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = segment_keys.cbegin();

  // Prepare input data
  value_t default_constant{};
  init_default_constant(default_constant);
  auto value_it = cuda::constant_iterator(default_constant);

  using op_t = cuda::std::plus<>;

  // Prepare verification data
  using accum_t = cuda::std::__accumulator_t<op_t, value_t, output_t>;
  c2h::host_vector<output_t> expected_result(num_segments);
  compute_segmented_problem_reference(value_it, segment_offsets, op_t{}, accum_t{}, expected_result.begin());
  c2h::host_vector<key_t> expected_keys = compute_unique_keys_reference(segment_keys);

  // Run test
  c2h::device_vector<offset_t> num_unique_keys(1);
  c2h::device_vector<key_t> out_unique_keys(num_segments);
  c2h::device_vector<output_t> out_result(num_segments);
  auto d_result_out_it = thrust::raw_pointer_cast(out_result.data());
  auto d_keys_out_it   = out_unique_keys.begin();
  device_reduce_by_key(
    d_keys_it,
    d_keys_out_it,
    value_it,
    d_result_out_it,
    thrust::raw_pointer_cast(num_unique_keys.data()),
    op_t{},
    num_items);

  // Verify result
  REQUIRE(expected_result == out_result);
}

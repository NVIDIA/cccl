// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_sum);

// %PARAM% TEST_LAUNCH lid 0:1

// List of types to test
using custom_t           = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;
using iterator_type_list = c2h::type_list<type_pair<custom_t>, type_pair<std::int64_t>>;
using offsets            = c2h::type_list<std::int32_t, std::uint32_t>;

C2H_TEST("Device segmented reduce works with fancy input iterators", "[reduce][device]", iterator_type_list, offsets)
{
  using type_pair_t = typename c2h::get<0, TestType>;
  using item_t      = typename type_pair_t::input_t;
  using output_t    = typename type_pair_t::output_t;
  using offset_t    = typename c2h::get<1, TestType>;

  constexpr int min_items = 1;
  constexpr int max_items = 1000000;

  // Number of items
  const int num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  INFO("Test num_items: " << num_items);

  // Range of segment sizes to generate
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  auto d_offsets_it           = thrust::raw_pointer_cast(segment_offsets.data());

  // Prepare input data
  item_t default_constant{};
  init_default_constant(default_constant);
  auto in_it = thrust::make_constant_iterator(default_constant);

  using op_t   = cuda::std::plus<>;
  using init_t = output_t;

  // Binary reduction operator
  auto reduction_op = op_t{};

  // Prepare verification data
  using accum_t = cuda::std::__accumulator_t<op_t, item_t, init_t>;
  c2h::host_vector<output_t> expected_result(num_segments);
  compute_segmented_problem_reference(in_it, segment_offsets, reduction_op, accum_t{}, expected_result.begin());

  // Run test
  c2h::device_vector<output_t> out_result(num_segments);
  auto d_out_it = thrust::raw_pointer_cast(out_result.data());
  device_segmented_reduce(in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1, reduction_op, init_t{});

  // Verify result
  REQUIRE(expected_result == out_result);
}

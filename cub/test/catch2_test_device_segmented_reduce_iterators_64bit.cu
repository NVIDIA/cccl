// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cstdint>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_sum);

// %PARAM% TEST_LAUNCH lid 0:1

// List of types to test
using offsets = c2h::type_list<std::ptrdiff_t, std::size_t>;

C2H_TEST("Device segmented reduce works with fancy input iterators and 64-bit offsets", "[reduce][device]", offsets)
{
  using offset_t = typename c2h::get<0, TestType>;
  using op_t     = cuda::std::plus<>;

  constexpr offset_t offset_zero           = 0;
  constexpr offset_t offset_one            = 1;
  constexpr offset_t iterator_value        = 2;
  constexpr offset_t min_items_per_segment = offset_one << 31;
  constexpr offset_t max_items_per_segment = offset_one << 33;

  constexpr int num_segments = 2;

  // generate individual segment lengths and store cumulative sum in segment_offsets
  const offset_t num_items_in_first_segment =
    GENERATE_COPY(take(2, random(min_items_per_segment, max_items_per_segment)));
  const offset_t num_items_in_second_segment =
    GENERATE_COPY(take(2, random(min_items_per_segment, max_items_per_segment)));
  c2h::device_vector<offset_t> segment_offsets = {
    offset_zero, num_items_in_first_segment, num_items_in_first_segment + num_items_in_second_segment};

  // store expected result and initialize device output container
  c2h::host_vector<offset_t> expected_result = {
    iterator_value * num_items_in_first_segment, iterator_value * num_items_in_second_segment};
  c2h::device_vector<offset_t> device_result(num_segments);

  // prepare device iterators
  auto in_it        = thrust::make_constant_iterator(iterator_value);
  auto d_offsets_it = thrust::raw_pointer_cast(segment_offsets.data());
  auto d_out_it     = thrust::raw_pointer_cast(device_result.data());

  // reduce
  device_segmented_reduce(in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1, op_t{}, offset_t{});

  // verify result
  REQUIRE(expected_result == device_result);
}

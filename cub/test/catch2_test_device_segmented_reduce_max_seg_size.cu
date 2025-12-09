// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/guarantee.h>
#include <cuda/__execution/max_segment_size.h>
#include <cuda/std/__execution/env.h>

#include "catch2_test_device_reduce.cuh"
#include <c2h/catch2_test_helper.h>

using full_type_list = c2h::type_list<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t>;

using offsets = c2h::type_list<std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;

C2H_TEST("Device segmented reduce works with static and dynamic max segment sizes",
         "[segmented][reduce][device]",
         full_type_list,
         offsets)
{
  using input_t  = typename c2h::get<0, TestType>;
  using output_t = input_t;
  using offset_t = typename c2h::get<1, TestType>;

  constexpr int min_items = 1;
  constexpr int max_items = 10000;

  // Number of items
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const int num_items = GENERATE_COPY(
    take(c2h::adjust_seed_count(2), random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  INFO("Test num_items: " << num_items);

  const cuda::execution::max_segment_size max_seg_size = GENERATE(
    5, // dynamic segment sizes
    10,
    100,
    1000,
    10000,
    cuda::execution::max_segment_size<5>{}, // static segment sizes
    cuda::execution::max_segment_size<10>{},
    cuda::execution::max_segment_size<100>{},
    cuda::execution::max_segment_size<1000>{},
    cuda::execution::max_segment_size<10000>{});

  // Range of segment sizes to generate
  // Note that the segment range [0, 1] may also include one last segment with more than 1 items
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{0, 1}, {1, max_seg_size}, {max_seg_size, max_seg_size}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  auto d_offsets_it           = thrust::raw_pointer_cast(segment_offsets.data());

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  c2h::gen(C2H_SEED(2), in_items);
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  SECTION("sum")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, output_t>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(in_items, segment_offsets, op_t{}, accum_t{}, expected_result.begin());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = unwrap_it(thrust::raw_pointer_cast(out_result.data()));

    auto g_env = cuda::execution::guarantee(max_seg_size);

    auto error =
      cub::DeviceSegmentedReduce::Sum(d_in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1, g_env);

    // Verify result
    REQUIRE(expected_result == out_result);
    REQUIRE(error == cudaSuccess);
  }
}

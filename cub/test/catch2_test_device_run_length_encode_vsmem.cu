// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/vector_generators.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::Encode, run_length_encode);

// %PARAM% TEST_LAUNCH lid 0:1:2

// Both keys are too large for the lookahead path, so they take the streaming one. The 264-byte key
// makes the agent fall back to a smaller block size; the 520-byte key also needs 130 KiB of vsmem.
using huge_key_list = c2h::type_list<c2h::custom_type_t<c2h::equal_comparable_t, c2h::huge_data<256>::type>,
                                     c2h::custom_type_t<c2h::equal_comparable_t, c2h::huge_data<512>::type>>;

CUB_TEST("DeviceRunLengthEncode::Encode works with huge keys", "[device][run_length_encode]", CUB_SMALL, huge_key_list)
{
  using key_t    = typename c2h::get<0, TestType>;
  using length_t = std::int32_t;
  using offset_t = std::uint32_t;

  const offset_t num_items = GENERATE_COPY(take(2, random(1, 10000)));

  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  const c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);

  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);

  // Expected runs: one per segment, the key at its start and its length
  const c2h::host_vector<offset_t> h_segment_offsets = segment_offsets;
  const c2h::host_vector<key_t> h_segment_keys       = segment_keys;
  c2h::host_vector<key_t> expected_keys(num_segments);
  c2h::host_vector<length_t> expected_lengths(num_segments);
  for (offset_t i = 0; i < num_segments; ++i)
  {
    expected_keys[i]    = h_segment_keys[h_segment_offsets[i]];
    expected_lengths[i] = static_cast<length_t>(h_segment_offsets[i + 1] - h_segment_offsets[i]);
  }

  c2h::device_vector<key_t> out_unique_keys(num_segments);
  c2h::device_vector<length_t> out_lengths(num_segments);
  c2h::device_vector<offset_t> out_num_runs(1);

  run_length_encode(
    thrust::raw_pointer_cast(segment_keys.data()),
    thrust::raw_pointer_cast(out_unique_keys.data()),
    thrust::raw_pointer_cast(out_lengths.data()),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  REQUIRE(num_segments == out_num_runs[0]);
  REQUIRE(expected_keys == out_unique_keys);
  REQUIRE(expected_lengths == out_lengths);
}

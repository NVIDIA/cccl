// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <cstdint>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/custom_type.h>
#include <c2h/generators.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanByKey, device_inclusive_scan_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScanByKey, device_exclusive_scan_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

template <std::size_t Size>
using huge_value_t = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::huge_data<Size>::type>;

template <std::size_t Size>
using huge_key_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::huge_data<Size>::type>;

using key_value_types = c2h::type_list<
  // Types large enough to dispatch to the fallback policy
  c2h::pair<std::int32_t, huge_value_t<256>>,
  c2h::pair<huge_key_t<128>, std::uint32_t>,
  // Type large enough to require virtual shared memory
  c2h::pair<huge_key_t<512>, std::uint32_t>>;

CUB_TEST(
  "Device scan-by-key works with huge keys and values", "[by_key][scan][vsmem][device]", CUB_SMALL, key_value_types)
{
  using key_t    = typename c2h::first<c2h::get<0, TestType>>;
  using value_t  = typename c2h::second<c2h::get<0, TestType>>;
  using offset_t = std::uint32_t;
  using op_t     = cuda::std::plus<>;
  using eq_op_t  = cuda::std::equal_to<>;

  const offset_t num_items = GENERATE_COPY(take(2, random(1, 10000)));

  // Range of segment sizes to generate (a segment is a series of consecutive equal keys)
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  const c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  c2h::device_vector<key_t> segment_keys(num_items, thrust::default_init);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = thrust::raw_pointer_cast(segment_keys.data());

  c2h::device_vector<value_t> in_values(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), in_values);
  auto d_values_it = thrust::raw_pointer_cast(in_values.data());

  c2h::device_vector<value_t> out_values(num_items, thrust::default_init);
  auto d_values_out_it = thrust::raw_pointer_cast(out_values.data());

  SECTION("inclusive scan")
  {
    c2h::host_vector<value_t> expected_result(num_items, thrust::default_init);
    compute_inclusive_scan_by_key_reference(in_values, segment_keys, expected_result.begin(), op_t{}, eq_op_t{});

    device_inclusive_scan_by_key(d_keys_it, d_values_it, d_values_out_it, op_t{}, num_items, eq_op_t{});

    REQUIRE(expected_result == out_values);
  }

  SECTION("exclusive scan")
  {
    c2h::host_vector<value_t> expected_result(num_items, thrust::default_init);
    compute_exclusive_scan_by_key_reference(
      in_values, segment_keys, expected_result.begin(), op_t{}, eq_op_t{}, value_t{});

    device_exclusive_scan_by_key(d_keys_it, d_values_it, d_values_out_it, op_t{}, value_t{}, num_items, eq_op_t{});

    REQUIRE(expected_result == out_values);
  }
}

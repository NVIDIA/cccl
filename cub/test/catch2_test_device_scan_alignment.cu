// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <cuda/std/functional>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

// %PARAM% TEST_LAUNCH lid 0

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// We cover types of various sizes smaller than 16 byte
using value_types = c2h::type_list<uint8_t, uint16_t, uint32_t, uint64_t>;

C2H_TEST("Device scan works with all device interfaces", "[scan][device]", value_types)
{
  using input_t  = c2h::get<0, TestType>;
  using output_t = input_t;
  using offset_t = cuda::std::int32_t;
  using op_t     = cuda::std::plus<>;

  constexpr offset_t max_offset    = 64;
  constexpr offset_t max_num_items = 8192;

  const auto offset    = GENERATE_COPY(values({0, 1, 3, 4, 7, 8, 11, 12, 16}), take(3, random(0, max_offset)));
  const auto num_items = GENERATE_COPY(values({1, max_num_items}), take(64, random(0, max_num_items)));

  CAPTURE(num_items, offset);

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items + offset + 1, thrust::no_init);
  c2h::gen(C2H_SEED(1), in_items);
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  // Prepare verification data
  c2h::host_vector<input_t> host_items(in_items);
  c2h::host_vector<output_t> expected_result(num_items, thrust::no_init);

  // Compute verification data
  compute_inclusive_scan_reference(
    host_items.cbegin() + offset, host_items.cbegin() + offset + num_items, expected_result.begin(), op_t{}, 0);

  // Run test
  constexpr output_t out_sentinel_value = 123;
  c2h::device_vector<output_t> out_result(num_items + offset + 1, out_sentinel_value);
  auto d_out_it = thrust::raw_pointer_cast(out_result.data());
  device_inclusive_scan(unwrap_it(d_in_it + offset), unwrap_it(d_out_it + offset), op_t{}, num_items);

  c2h::host_vector<output_t> out_result_vec(num_items);
  thrust::copy_n(out_result.begin() + offset, num_items, out_result_vec.begin());

  REQUIRE_THAT_QUIET(out_result_vec, Equals(expected_result));

  const output_t out_sentinel = out_result[offset + num_items];
  REQUIRE(out_sentinel == out_sentinel_value);
}

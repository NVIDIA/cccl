// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <thrust/copy.h>

#include <cuda/std/functional>

#include <cstdint>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);

// trivially constructible types to allow uninitialized thrust vector
using types = c2h::type_list<std::int32_t, std::int64_t>;

// The last element must not be read in an exclusive scan, to be confirmed by compute-sanitizer initcheck
CUB_TEST("Device exclusive scan ignores last input element", "[scan][device]", CUB_SMALL, types)
{
  using type     = c2h::get<0, TestType>;
  using offset_t = std::int32_t;
  using op_t     = ::cuda::std::plus<>;

  const offset_t initialized_size =
    GENERATE_COPY(values({0, 1, 2, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097}), take(3, random(1, 1'000'000)));
  const offset_t size = initialized_size + 1;
  CAPTURE(size, c2h::type_name<type>());

  c2h::device_vector<type> input(initialized_size);
  c2h::gen(C2H_SEED(1), input);
  // Initialize all but the last value of the device vector
  c2h::device_vector<type> data(size, thrust::no_init);
  thrust::copy(c2h::device_policy, input.cbegin(), input.cend(), data.begin());
  auto d_data = thrust::raw_pointer_cast(data.data());

  c2h::host_vector<type> host_input(input);
  host_input.push_back(type{}); // the last value doesn't matter here either
  c2h::host_vector<type> expected(size);

  type init_value{};
  compute_exclusive_scan_reference(host_input.cbegin(), host_input.cend(), expected.begin(), init_value, op_t{});

  device_exclusive_scan(d_data, d_data, op_t{}, init_value, size);

  REQUIRE_THAT_QUIET(expected, Equals(data));
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBoundSortedValues, device_lower_bound_sorted_values);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBoundSortedValues, device_upper_bound_sorted_values);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("Device LowerBoundSortedValues works with default environment", "[find][device][binary-search]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  auto error = cub::DeviceFind::LowerBoundSortedValues(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<int> expected = {0, 2, 2, 4};
  REQUIRE(d_output == expected);
}

TEST_CASE("Device UpperBoundSortedValues works with default environment", "[find][device][binary-search]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  auto error = cub::DeviceFind::UpperBoundSortedValues(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

#endif

C2H_TEST("Device LowerBoundSortedValues uses environment", "[find][device][binary-search]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::LowerBoundSortedValues(
      nullptr,
      expected_bytes_allocated,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_lower_bound_sorted_values(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    env);

  c2h::device_vector<int> expected = {0, 2, 2, 4};
  REQUIRE(d_output == expected);
}

C2H_TEST("Device UpperBoundSortedValues uses environment", "[find][device][binary-search]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::UpperBoundSortedValues(
      nullptr,
      expected_bytes_allocated,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_upper_bound_sorted_values(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    env);

  c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

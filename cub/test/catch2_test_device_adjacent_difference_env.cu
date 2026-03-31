// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceAdjacentDifference::SubtractLeftCopy, device_adjacent_difference_subtract_left_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceAdjacentDifference::SubtractLeft, device_adjacent_difference_subtract_left);
DECLARE_LAUNCH_WRAPPER(cub::DeviceAdjacentDifference::SubtractRightCopy, device_adjacent_difference_subtract_right_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceAdjacentDifference::SubtractRight, device_adjacent_difference_subtract_right);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("Device adjacent difference subtract left copy works with default environment",
          "[adjacent_difference][device]")
{
  auto input  = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = c2h::device_vector<int>(8);

  REQUIRE(cudaSuccess
          == cub::DeviceAdjacentDifference::SubtractLeftCopy(
            input.begin(), output.begin(), input.size(), cuda::std::minus{}));

  c2h::device_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  REQUIRE(output == expected);
}

TEST_CASE("Device adjacent difference subtract left works with default environment", "[adjacent_difference][device]")
{
  auto data = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractLeft(data.begin(), data.size(), cuda::std::minus{}));

  c2h::device_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  REQUIRE(data == expected);
}

TEST_CASE("Device adjacent difference subtract right copy works with default environment",
          "[adjacent_difference][device]")
{
  auto input  = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = c2h::device_vector<int>(8);

  REQUIRE(cudaSuccess
          == cub::DeviceAdjacentDifference::SubtractRightCopy(
            input.begin(), output.begin(), input.size(), cuda::std::minus{}));

  c2h::device_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  REQUIRE(output == expected);
}

TEST_CASE("Device adjacent difference subtract right works with default environment", "[adjacent_difference][device]")
{
  auto data = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  REQUIRE(cudaSuccess == cub::DeviceAdjacentDifference::SubtractRight(data.begin(), data.size(), cuda::std::minus{}));

  c2h::device_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  REQUIRE(data == expected);
}

#endif

C2H_TEST("Device adjacent difference subtract left copy uses environment", "[adjacent_difference][device]")
{
  auto input  = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = c2h::device_vector<int>(8);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceAdjacentDifference::SubtractLeftCopy(
            nullptr, expected_bytes_allocated, input.begin(), output.begin(), input.size(), cuda::std::minus{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_adjacent_difference_subtract_left_copy(input.begin(), output.begin(), input.size(), cuda::std::minus{}, env);

  c2h::device_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  REQUIRE(output == expected);
}

C2H_TEST("Device adjacent difference subtract left uses environment", "[adjacent_difference][device]")
{
  auto data = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceAdjacentDifference::SubtractLeft(
            nullptr, expected_bytes_allocated, data.begin(), data.size(), cuda::std::minus{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_adjacent_difference_subtract_left(data.begin(), data.size(), cuda::std::minus{}, env);

  c2h::device_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  REQUIRE(data == expected);
}

C2H_TEST("Device adjacent difference subtract right copy uses environment", "[adjacent_difference][device]")
{
  auto input  = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = c2h::device_vector<int>(8);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceAdjacentDifference::SubtractRightCopy(
            nullptr, expected_bytes_allocated, input.begin(), output.begin(), input.size(), cuda::std::minus{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_adjacent_difference_subtract_right_copy(input.begin(), output.begin(), input.size(), cuda::std::minus{}, env);

  c2h::device_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  REQUIRE(output == expected);
}

C2H_TEST("Device adjacent difference subtract right uses environment", "[adjacent_difference][device]")
{
  auto data = c2h::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceAdjacentDifference::SubtractRight(
            nullptr, expected_bytes_allocated, data.begin(), data.size(), cuda::std::minus{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_adjacent_difference_subtract_right(data.begin(), data.size(), cuda::std::minus{}, env);

  c2h::device_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  REQUIRE(data == expected);
}

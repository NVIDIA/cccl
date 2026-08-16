// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinMax, device_arg_minmax);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinLastMax, device_arg_minlastmax);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device ArgMinMax works with default environment", "[reduce][device]", CUB_SMALL)
{
  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<float>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST_CASE("Device ArgMinLastMax works with default environment", "[reduce][device]", CUB_SMALL)
{
  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<float>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMinLastMax(
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

#endif

CUB_TEST("Device ArgMinMax uses environment", "[reduce][device]", CUB_SMALL)
{
  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<float>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  const auto n = static_cast<::cuda::std::int64_t>(input.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMinMax(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_out.begin(),
      min_index.begin(),
      max_out.begin(),
      max_index.begin(),
      n));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_minmax(input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, env);

  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST("Device ArgMinLastMax uses environment", "[reduce][device]", CUB_SMALL)
{
  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<float>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  const auto n = static_cast<::cuda::std::int64_t>(input.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMinLastMax(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_out.begin(),
      min_index.begin(),
      max_out.begin(),
      max_index.begin(),
      n));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_minlastmax(input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, env);

  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST("Device ArgMinMax with compare_op uses environment", "[reduce][device]", CUB_SMALL)
{
  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<float>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  const auto n = static_cast<::cuda::std::int64_t>(input.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMinMax(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_out.begin(),
      min_index.begin(),
      max_out.begin(),
      max_index.begin(),
      n,
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_minmax(
    input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, cuda::std::less{}, env);

  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST("Device ArgMinLastMax with compare_op uses environment", "[reduce][device]", CUB_SMALL)
{
  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<float>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  const auto n = static_cast<::cuda::std::int64_t>(input.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMinLastMax(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_out.begin(),
      min_index.begin(),
      max_out.begin(),
      max_index.begin(),
      n,
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_minlastmax(
    input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, cuda::std::less{}, env);

  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

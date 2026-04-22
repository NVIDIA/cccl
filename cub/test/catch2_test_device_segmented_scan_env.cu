// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_segmented_exclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_segmented_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_segmented_inclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_segmented_inclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_segmented_inclusive_scan_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("Device segmented exclusive sum works with default environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments));

  thrust::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented exclusive scan works with default environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100));

  thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented inclusive sum works with default environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments));

  thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented inclusive scan works with default environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}));

  thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented inclusive scan init works with default environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100));

  thrust::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
}

#endif

C2H_TEST("Device segmented exclusive sum uses environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_sum(d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, env);

  thrust::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented exclusive scan uses environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_offsets_it,
      d_offsets_it + 1,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_scan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100, env);

  thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented inclusive sum uses environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
      nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_sum(d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, env);

  thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented inclusive scan uses environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_offsets_it,
      d_offsets_it + 1,
      num_segments,
      ::cuda::std::plus<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, env);

  thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented inclusive scan init uses environment", "[segmented_scan][device]")
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_offsets_it,
      d_offsets_it + 1,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan_init(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100, env);

  thrust::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
}

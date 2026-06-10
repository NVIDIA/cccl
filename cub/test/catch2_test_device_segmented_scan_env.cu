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
#include <cuda/__execution/tune.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Test data (separate input/output offsets):
// Input data: {1,2,3,4,5,6,7,8}  - 3 segments of sizes 3, 2, 3 at input offsets {0,3,5,8}
// Output layout: 10 slots with padding at positions 3 and 6; output begin offsets {0,4,7}

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

TEST_CASE("Device segmented exclusive sum with separate offsets works with default environment",
          "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments));

  thrust::device_vector<int> expected{0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented exclusive scan with separate offsets works with default environment",
          "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
      d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, 100));

  thrust::device_vector<int> expected{100, 101, 103, sentinel, 100, 104, sentinel, 100, 106, 113};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented inclusive sum with separate offsets works with default environment",
          "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments));

  thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented inclusive scan with separate offsets works with default environment",
          "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
      d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}));

  thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device segmented inclusive scan init with separate offsets works with default environment",
          "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
      d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, 100));

  thrust::device_vector<int> expected{101, 103, 106, sentinel, 104, 109, sentinel, 106, 113, 121};
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

C2H_TEST("Device segmented exclusive sum with separate offsets uses environment", "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_sum(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, env);

  thrust::device_vector<int> expected{0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented exclusive scan with separate offsets uses environment", "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_scan(
    d_in.begin(),
    d_out.begin(),
    d_in_off_it,
    d_in_off_it + 1,
    d_out_off_it,
    num_segments,
    ::cuda::std::plus<>{},
    100,
    env);

  thrust::device_vector<int> expected{100, 101, 103, sentinel, 100, 104, sentinel, 100, 106, 113};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented inclusive sum with separate offsets uses environment", "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_sum(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, env);

  thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented inclusive scan with separate offsets uses environment", "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments,
      ::cuda::std::plus<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, env);

  thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

C2H_TEST("Device segmented inclusive scan init with separate offsets uses environment", "[segmented_scan][device]")
{
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan_init(
    d_in.begin(),
    d_out.begin(),
    d_in_off_it,
    d_in_off_it + 1,
    d_out_off_it,
    num_segments,
    ::cuda::std::plus<>{},
    100,
    env);

  thrust::device_vector<int> expected{101, 103, 106, sentinel, 104, 109, sentinel, 106, 113, 121};
  REQUIRE(d_out == expected);
}

#if TEST_LAUNCH != 1

// A policy selector that forces a specific block size. We deliberately use direct block load/store so that the chosen
// block size is valid for any of the values exercised below.
template <unsigned int BlockThreads>
struct segmented_scan_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::detail::segmented_scan::segmented_scan_policy
  {
    return cub::detail::segmented_scan::segmented_scan_policy{cub::detail::segmented_scan::block_segmented_scan_policy{
      static_cast<int>(BlockThreads),
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_DIRECT,
      cub::BLOCK_SCAN_WARP_SCANS,
      512}};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

// The "Sum" APIs do not take a user operator, so we extract the launched block size through a custom input iterator
// (block_size_extracting_constant_iterator). The "Scan" APIs take a user scan operator, so we wrap a plus<> in
// block_size_extracting_op which both records the block size and computes the expected result.

C2H_TEST("Device segmented exclusive sum can be tuned", "[segmented_scan][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(9);
  c2h::device_vector<unsigned int> d_block_size(1);
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_exclusive_sum(d_in, d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device segmented inclusive sum can be tuned", "[segmented_scan][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_out(9);
  c2h::device_vector<unsigned int> d_block_size(1);
  auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_inclusive_sum(d_in, d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device segmented exclusive scan can be tuned", "[segmented_scan][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_exclusive_scan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);

  thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device segmented inclusive scan can be tuned", "[segmented_scan][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_inclusive_scan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, env);

  thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device segmented inclusive scan init can be tuned", "[segmented_scan][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_inclusive_scan_init(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);

  thrust::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

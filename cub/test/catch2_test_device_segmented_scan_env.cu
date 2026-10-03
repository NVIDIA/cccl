// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>
#include <cuda/std/limits>

#include <sstream>
#include <vector>

#include "block_size_extracting_helpers.h"
#include "catch2_test_device_segmented_scan_schedules.cuh"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_segmented_exclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_segmented_exclusive_scan);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_segmented_inclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_segmented_inclusive_scan);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_segmented_inclusive_scan_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

// Test data (separate input/output offsets):
// Input data: {1,2,3,4,5,6,7,8}  - 3 segments of sizes 3, 2, 3 at input offsets {0,3,5,8}
// Output layout: 10 slots with padding at positions 3 and 6; output begin offsets {0,4,7}

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device segmented exclusive sum works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments));

  const thrust::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented exclusive scan works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100));

  const thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive sum works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments));

  const thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented scan rejects a negative num_items", "[segmented_scan][device]", CUB_SMALL)
{
  thrust::device_vector<int> d_offsets{0, 2};
  thrust::device_vector<int> d_in{1, 1};
  thrust::device_vector<int> d_out(2);
  const auto env = stdexec::env{cub::segmented_scan_num_items(-1), cub::segmented_scan_load_balancing};
  REQUIRE(cudaErrorInvalidValue
          == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            d_in.begin(), d_out.begin(), d_offsets.begin(), d_offsets.begin() + 1, 1, env));
  REQUIRE(
    cudaErrorInvalidValue
    == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
      d_in.begin(),
      d_out.begin(),
      d_offsets.begin(),
      d_offsets.begin() + 1,
      1,
      stdexec::env{cub::segmented_scan_num_items(-1)}));
}

CUB_TEST_CASE("Device segmented inclusive scan works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}));

  const thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive scan init works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
            d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100));

  const thrust::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented exclusive sum with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
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

  const thrust::device_vector<int> expected{0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented exclusive scan with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
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

  const thrust::device_vector<int> expected{100, 101, 103, sentinel, 100, 104, sentinel, 100, 106, 113};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive sum with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
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

  const thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive scan with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
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

  const thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive scan init with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
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

  const thrust::device_vector<int> expected{101, 103, 106, sentinel, 104, 109, sentinel, 106, 113, 121};
  REQUIRE(d_out == expected);
}

#endif

CUB_TEST("Device segmented exclusive sum uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented exclusive scan uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented inclusive sum uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented inclusive scan uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented inclusive scan init uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented exclusive sum with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented exclusive scan with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{100, 101, 103, sentinel, 100, 104, sentinel, 100, 106, 113};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented inclusive sum with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented inclusive scan with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
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

  const thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device segmented inclusive scan init with separate offsets uses environment",
         "[segmented_scan][device]",
         CUB_SMALL)
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

  const thrust::device_vector<int> expected{101, 103, 106, sentinel, 104, 109, sentinel, 106, 113, 121};
  REQUIRE(d_out == expected);
}

using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t>;

// Segments placed out of order in the input and in the output, with gaps that hold a canary between them.
CUB_TEST("Device segmented scan with schedule properties scans out-of-order segments with gaps",
         "[segmented_scan][device]",
         CUB_SMALL,
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;

  constexpr int canary          = -1;
  constexpr int init            = 7;
  constexpr offset_t input_gap  = 3;
  constexpr offset_t output_gap = 5;

  // Segments longer and shorter than a tile, and empty ones. The mean of 97 items lets several share a block.
  std::vector<offset_t> sizes{3, 0, 1500, 7, 0, 0, 129, 512, 1, 40, 2100, 5, 64, 0, 511, 513, 2};
  for (offset_t i = 0; i < 40; ++i)
  {
    sizes.push_back(i % 8);
  }
  const auto num_segments = static_cast<::cuda::std::int64_t>(sizes.size());

  // Segment (slot * 37) % num_segments takes the slot-th place, in the input and in the output.
  c2h::host_vector<offset_t> h_begins(num_segments);
  c2h::host_vector<offset_t> h_ends(num_segments);
  c2h::host_vector<offset_t> h_out_begins(num_segments);
  offset_t num_input             = input_gap;
  offset_t num_output            = output_gap;
  ::cuda::std::int64_t num_items = 0;
  for (::cuda::std::int64_t slot = 0; slot < num_segments; ++slot)
  {
    const auto segment    = static_cast<size_t>((slot * 37) % num_segments);
    h_begins[segment]     = num_input;
    h_ends[segment]       = num_input + sizes[segment];
    h_out_begins[segment] = num_output;
    num_input += sizes[segment] + input_gap;
    num_output += sizes[segment] + output_gap;
    num_items += sizes[segment];
  }
  // Segment 13 holds no items. A segment whose end precedes its begin is empty too.
  h_ends[13] = h_begins[13] - 1;

  const c2h::device_vector<offset_t> d_begins     = h_begins;
  const c2h::device_vector<offset_t> d_ends       = h_ends;
  const c2h::device_vector<offset_t> d_out_begins = h_out_begins;
  const offset_t* d_begins_it                     = thrust::raw_pointer_cast(d_begins.data());
  const offset_t* d_ends_it                       = thrust::raw_pointer_cast(d_ends.data());
  const offset_t* d_out_begins_it                 = thrust::raw_pointer_cast(d_out_begins.data());

  c2h::device_vector<int> d_in(num_input);
  c2h::gen(C2H_SEED(1), d_in, -100, 100);
  const c2h::host_vector<int> h_in = d_in;

  // Scans every segment of h_in into expected, starting at the segment's entry of h_dest_begins.
  const auto scan_segments =
    [&](c2h::host_vector<int>& expected, const c2h::host_vector<offset_t>& h_dest_begins, bool inclusive, int first) {
      for (size_t segment = 0; segment < sizes.size(); ++segment)
      {
        int prefix = first;
        for (offset_t item = 0; item < sizes[segment]; ++item)
        {
          const int value                         = h_in[h_begins[segment] + item];
          expected[h_dest_begins[segment] + item] = inclusive ? prefix + value : prefix;
          prefix += value;
        }
      }
    };

  const auto num_items_env     = stdexec::env{cub::segmented_scan_num_items(num_items)};
  const auto load_balanced_env = stdexec::env{cub::segmented_scan_load_balancing};
  const auto both_env = stdexec::env{cub::segmented_scan_num_items(num_items), cub::segmented_scan_load_balancing};

  SECTION("inclusive sum")
  {
    c2h::host_vector<int> expected(num_output, canary);
    scan_segments(expected, h_out_begins, true, 0);
    c2h::device_vector<int> d_out(num_output);

    const auto run = [&](const auto& schedule_env) {
      const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
        schedule_env,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_begins_it,
        d_ends_it,
        d_out_begins_it,
        ::cuda::std::plus<>{},
        cub::NullType{});
      thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), canary);
      device_segmented_inclusive_sum(
        d_in.begin(),
        d_out.begin(),
        d_begins_it,
        d_ends_it,
        d_out_begins_it,
        num_segments,
        stdexec::env{expected_allocation_size(bytes), schedule_env});
      REQUIRE(c2h::host_vector<int>(d_out) == expected);
    };
    run(num_items_env);
    run(load_balanced_env);
    run(both_env);
  }

  SECTION("exclusive scan in place")
  {
    // The gaps keep their input values.
    c2h::host_vector<int> expected = h_in;
    scan_segments(expected, h_begins, false, init);
    c2h::device_vector<int> d_data(num_input);

    const auto run = [&](const auto& schedule_env) {
      const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, true>(
        schedule_env,
        d_data.begin(),
        d_data.begin(),
        num_segments,
        d_begins_it,
        d_ends_it,
        d_begins_it,
        ::cuda::std::plus<>{},
        cub::detail::InputValue<int>{init});
      d_data = d_in;
      device_segmented_exclusive_scan(
        d_data.begin(),
        d_data.begin(),
        d_begins_it,
        d_ends_it,
        num_segments,
        ::cuda::std::plus<>{},
        init,
        stdexec::env{expected_allocation_size(bytes), schedule_env});
      REQUIRE(c2h::host_vector<int>(d_data) == expected);
    };
    run(num_items_env);
    run(load_balanced_env);
    run(both_env);
  }
}

// The input at index base + k is k % 7 + 1.
template <typename OffsetT>
struct high_offset_value
{
  OffsetT base;

  __host__ __device__ ::cuda::std::int64_t operator()(OffsetT index) const
  {
    return static_cast<::cuda::std::int64_t>((index - base) % OffsetT{7} + OffsetT{1});
  }
};

CUB_TEST("Device segmented scan with load balancing scans segments near the offset maximum",
         "[segmented_scan][device]",
         CUB_SMALL,
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  using value_t  = ::cuda::std::int64_t;

  // Two segments of 1 and 63 items that begin within 100 of the offset maximum and write to the start of the output.
  const offset_t base = ::cuda::std::numeric_limits<offset_t>::max() - offset_t{100};
  const c2h::device_vector<offset_t> d_begins{base, base + offset_t{1}};
  const c2h::device_vector<offset_t> d_ends{base + offset_t{1}, base + offset_t{64}};
  const c2h::device_vector<offset_t> d_out_begins{0, 1};
  const offset_t* d_begins_it             = thrust::raw_pointer_cast(d_begins.data());
  const offset_t* d_ends_it               = thrust::raw_pointer_cast(d_ends.data());
  const offset_t* d_out_begins_it         = thrust::raw_pointer_cast(d_out_begins.data());
  const ::cuda::std::int64_t num_segments = 2;
  const value_t init                      = 7;
  const auto d_in = cuda::transform_iterator(cuda::counting_iterator(offset_t{0}), high_offset_value<offset_t>{base});
  c2h::device_vector<value_t> d_out(64, value_t{-1});

  c2h::host_vector<value_t> expected(64);
  value_t prefix = init;
  expected[0]    = prefix;
  for (int item = 1; item < 64; ++item)
  {
    expected[item] = prefix;
    prefix += item % 7 + 1;
  }

  const auto schedule_env = stdexec::env{cub::segmented_scan_load_balancing};
  const size_t bytes      = schedule_allocation_size<cub::ForceInclusive::No, false>(
    schedule_env,
    d_in,
    d_out.begin(),
    num_segments,
    d_begins_it,
    d_ends_it,
    d_out_begins_it,
    ::cuda::std::plus<>{},
    cub::detail::InputValue<value_t>{init});
  device_segmented_exclusive_scan(
    d_in,
    d_out.begin(),
    d_begins_it,
    d_ends_it,
    d_out_begins_it,
    num_segments,
    ::cuda::std::plus<>{},
    init,
    stdexec::env{expected_allocation_size(bytes), schedule_env});
  REQUIRE(c2h::host_vector<value_t>(d_out) == expected);
}

CUB_TEST("Device segmented scan with schedule properties skips runs of empty segments",
         "[segmented_scan][device]",
         CUB_SMALL)
{
  using offset_t = ::cuda::std::int64_t;

  constexpr int canary = -1;
  constexpr int init   = 7;

  // Eight groups of 1024 items. Each starts with a run of empty segments and holds a second run after 301 items, so
  // runs of several lengths begin both at the first item of a tile and inside one. Every run is followed by a one-item
  // segment. The first segment and the last segment are empty.
  const int runs[] = {3, 1, 2, 31, 32, 33, 10'000, 5};
  std::vector<offset_t> h_offsets_vec{0};
  const auto push = [&](int count, offset_t size) {
    for (int i = 0; i < count; ++i)
    {
      h_offsets_vec.push_back(h_offsets_vec.back() + size);
    }
  };
  for (int group = 0; group < 8; ++group)
  {
    push(runs[group], 0);
    push(1, 1);
    push(1, 300);
    push(runs[7 - group], 0);
    push(1, 1);
    push(1, 722);
  }
  push(33, 0);

  const c2h::host_vector<offset_t> h_offsets(h_offsets_vec.begin(), h_offsets_vec.end());
  const c2h::device_vector<offset_t> d_offsets = h_offsets;
  const offset_t* d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  const auto num_segments                      = static_cast<::cuda::std::int64_t>(h_offsets.size() - 1);
  const offset_t num_items                     = h_offsets_vec.back();

  c2h::device_vector<int> d_in(num_items);
  c2h::gen(C2H_SEED(1), d_in, -100, 100);
  const c2h::host_vector<int> h_in = d_in;

  const auto num_items_env     = stdexec::env{cub::segmented_scan_num_items(num_items)};
  const auto load_balanced_env = stdexec::env{cub::segmented_scan_load_balancing};
  const auto both_env = stdexec::env{cub::segmented_scan_num_items(num_items), cub::segmented_scan_load_balancing};

  SECTION("exclusive scan")
  {
    c2h::host_vector<int> expected(num_items);
    for (::cuda::std::int64_t segment = 0; segment < num_segments; ++segment)
    {
      int prefix = init;
      for (offset_t item = h_offsets[segment]; item < h_offsets[segment + 1]; ++item)
      {
        expected[item] = prefix;
        prefix += h_in[item];
      }
    }
    c2h::device_vector<int> d_out(num_items);

    const auto run = [&](const auto& schedule_env) {
      const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, true>(
        schedule_env,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        d_offsets_it,
        ::cuda::std::plus<>{},
        cub::detail::InputValue<int>{init});
      thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), canary);
      device_segmented_exclusive_scan(
        d_in.begin(),
        d_out.begin(),
        d_offsets_it,
        d_offsets_it + 1,
        num_segments,
        ::cuda::std::plus<>{},
        init,
        stdexec::env{expected_allocation_size(bytes), schedule_env});
      REQUIRE(c2h::host_vector<int>(d_out) == expected);
    };
    run(num_items_env);
    run(load_balanced_env);
    run(both_env);
  }

  SECTION("inclusive sum with gaps between output segments")
  {
    const auto num_output = static_cast<size_t>(num_items) + static_cast<size_t>(num_segments);
    c2h::host_vector<offset_t> h_out_offsets(num_segments);
    c2h::host_vector<int> expected(num_output, canary);
    for (::cuda::std::int64_t segment = 0; segment < num_segments; ++segment)
    {
      h_out_offsets[segment] = h_offsets[segment] + static_cast<offset_t>(segment);
      int sum                = 0;
      for (offset_t item = h_offsets[segment]; item < h_offsets[segment + 1]; ++item)
      {
        sum += h_in[item];
        expected[h_out_offsets[segment] + (item - h_offsets[segment])] = sum;
      }
    }
    const c2h::device_vector<offset_t> d_out_offsets = h_out_offsets;
    const offset_t* d_out_offsets_it                 = thrust::raw_pointer_cast(d_out_offsets.data());
    c2h::device_vector<int> d_out(num_output);

    const auto run = [&](const auto& schedule_env) {
      const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
        schedule_env,
        d_in.begin(),
        d_out.begin(),
        num_segments,
        d_offsets_it,
        d_offsets_it + 1,
        d_out_offsets_it,
        ::cuda::std::plus<>{},
        cub::NullType{});
      thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), canary);
      device_segmented_inclusive_sum(
        d_in.begin(),
        d_out.begin(),
        d_offsets_it,
        d_offsets_it + 1,
        d_out_offsets_it,
        num_segments,
        stdexec::env{expected_allocation_size(bytes), schedule_env});
      REQUIRE(c2h::host_vector<int>(d_out) == expected);
    };
    run(num_items_env);
    run(load_balanced_env);
    run(both_env);
  }
}

CUB_TEST("Device segmented scan with schedule properties handles zero segments", "[segmented_scan][device]", CUB_SMALL)
{
  const int canary                     = -1;
  thrust::device_vector<int> d_offsets = {0};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5};
  thrust::device_vector<int> d_out(d_in.size(), canary);
  const thrust::device_vector<int> expected(d_in.size(), canary);

  const auto run = [&](const auto& schedule_env) {
    device_segmented_inclusive_sum(
      d_in.begin(),
      d_out.begin(),
      d_offsets_it,
      d_offsets_it + 1,
      0,
      stdexec::env{expected_allocation_size(0), schedule_env});
    REQUIRE(d_out == expected);
  };
  run(stdexec::env{cub::segmented_scan_num_items(0)});
  run(stdexec::env{cub::segmented_scan_load_balancing});
  run(stdexec::env{cub::segmented_scan_num_items(0), cub::segmented_scan_load_balancing});
}

#if TEST_LAUNCH != 1

// A policy selector that forces a specific block size. We deliberately use direct block load/store so that the chosen
// block size is valid for any of the values exercised below.
template <unsigned int BlockThreads>
struct segmented_scan_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> cub::SegmentedScanPolicy
  {
    return cub::SegmentedScanPolicy{cub::SegmentedScanBlockPolicy{
      static_cast<int>(BlockThreads),
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_DIRECT,
      cub::BLOCK_SCAN_WARP_SCANS,
      512}};
  }
};

template <unsigned int BlockThreads>
struct segmented_scan_load_balanced_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::SegmentedScanLoadBalancedPolicy
  {
    return cub::SegmentedScanLoadBalancedPolicy{
      static_cast<int>(BlockThreads),
      1,
      cub::BLOCK_LOAD_DIRECT,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_DIRECT,
      cub::BLOCK_SCAN_WARP_SCANS,
      16,
      5};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

// The "Sum" APIs do not take a user operator, so we extract the launched block size through a custom input iterator
// (block_size_extracting_constant_iterator). The "Scan" APIs take a user scan operator, so we wrap a plus<> in
// block_size_extracting_op which both records the block size and computes the expected result.

CUB_TEST("Device segmented exclusive sum can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
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

CUB_TEST("Device segmented inclusive sum can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
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

CUB_TEST("Device segmented exclusive scan can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_exclusive_scan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);

  const thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device segmented inclusive scan can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_inclusive_scan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, env);

  const thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device segmented inclusive scan init can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  device_segmented_inclusive_scan_init(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);

  const thrust::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "Device segmented scan with schedule properties can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  constexpr unsigned int other_block_size  = target_block_size == 64 ? 128 : 64;
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};
  const thrust::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};

  SECTION("with both selectors")
  {
    const auto env = stdexec::env{
      cub::segmented_scan_load_balancing,
      cuda::execution::tune(segmented_scan_tuning<other_block_size>{},
                            segmented_scan_load_balanced_tuning<target_block_size>{})};
    device_segmented_exclusive_scan(
      d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);
    REQUIRE(d_out == expected);
    REQUIRE(d_block_size[0] == target_block_size);
  }

  SECTION("with only the load-balanced selector")
  {
    const auto env = stdexec::env{cub::segmented_scan_load_balancing,
                                  cuda::execution::tune(segmented_scan_load_balanced_tuning<target_block_size>{})};
    device_segmented_exclusive_scan(
      d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);
    REQUIRE(d_out == expected);
    REQUIRE(d_block_size[0] == target_block_size);
  }

  // The block selector does not tune load balancing; this checks that the combination is accepted.
  SECTION("with only the block selector")
  {
    const auto env = stdexec::env{
      cub::segmented_scan_load_balancing, cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};
    device_segmented_exclusive_scan(
      d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);
    REQUIRE(d_out == expected);
  }

  SECTION("with num_items and the block selector")
  {
    const auto env =
      stdexec::env{cub::segmented_scan_num_items(9), cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};
    device_segmented_exclusive_scan(
      d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);
    REQUIRE(d_out == expected);
    REQUIRE(d_block_size[0] == target_block_size);
  }
}

#endif // TEST_LAUNCH != 1

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test SegmentedScanPolicy properties", "[segmented_scan][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedScanPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedScanPolicy>);
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedScanBlockPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedScanBlockPolicy>);

  // aggregate init
  constexpr auto block1 = cub::SegmentedScanBlockPolicy{
    128,
    9,
    cub::BLOCK_LOAD_WARP_TRANSPOSE,
    cub::LOAD_DEFAULT,
    cub::BLOCK_STORE_WARP_TRANSPOSE,
    cub::BLOCK_SCAN_WARP_SCANS,
    512};
  constexpr auto p1 = cub::SegmentedScanPolicy{block1};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto block2 = cub::SegmentedScanBlockPolicy{
    .threads_per_block = 128,
    .items_per_thread  = 9,
    .load_algorithm    = cub::BLOCK_LOAD_WARP_TRANSPOSE,
    .load_modifier     = cub::LOAD_DEFAULT,
    .store_algorithm   = cub::BLOCK_STORE_WARP_TRANSPOSE,
    .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
    .max_segments      = 512};
  constexpr auto p2 = cub::SegmentedScanPolicy{.block = block2};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto block2 = block1;
  constexpr auto p2     = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(block1 == block2);
  STATIC_REQUIRE_FALSE(block1 != block2);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(block1)
          == "SegmentedScanBlockPolicy { .threads_per_block = 128, .items_per_thread = 9"
             ", .load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE, .load_modifier = LOAD_DEFAULT"
             ", .store_algorithm = BLOCK_STORE_WARP_TRANSPOSE, .scan_algorithm = BLOCK_SCAN_WARP_SCANS"
             ", .max_segments_per_block = 512 }");
  REQUIRE(to_string(p1)
          == "SegmentedScanPolicy { .block = SegmentedScanBlockPolicy { .threads_per_block = 128"
             ", .items_per_thread = 9, .load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE"
             ", .load_modifier = LOAD_DEFAULT, .store_algorithm = BLOCK_STORE_WARP_TRANSPOSE"
             ", .scan_algorithm = BLOCK_SCAN_WARP_SCANS, .max_segments_per_block = 512 } }");
}

CUB_TEST("Test SegmentedScanLoadBalancedPolicy properties", "[segmented_scan][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::SegmentedScanLoadBalancedPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::SegmentedScanLoadBalancedPolicy>);

  // aggregate init
  constexpr auto p1 = cub::SegmentedScanLoadBalancedPolicy{
    128,
    4,
    cub::BLOCK_LOAD_WARP_TRANSPOSE,
    cub::LOAD_DEFAULT,
    cub::BLOCK_STORE_WARP_TRANSPOSE,
    cub::BLOCK_SCAN_WARP_SCANS,
    16,
    5};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::SegmentedScanLoadBalancedPolicy{
    .threads_per_block       = 128,
    .items_per_thread        = 4,
    .load_algorithm          = cub::BLOCK_LOAD_WARP_TRANSPOSE,
    .load_modifier           = cub::LOAD_DEFAULT,
    .store_algorithm         = cub::BLOCK_STORE_WARP_TRANSPOSE,
    .scan_algorithm          = cub::BLOCK_SCAN_WARP_SCANS,
    .min_tiles_per_block     = 16,
    .max_subscription_factor = 5};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  std::ostringstream os;
  os << p1;
  REQUIRE(os.str()
          == "SegmentedScanLoadBalancedPolicy { .threads_per_block = 128, .items_per_thread = 4"
             ", .load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE, .load_modifier = LOAD_DEFAULT"
             ", .store_algorithm = BLOCK_STORE_WARP_TRANSPOSE, .scan_algorithm = BLOCK_SCAN_WARP_SCANS"
             ", .min_tiles_per_block = 16, .max_subscription_factor = 5 }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)

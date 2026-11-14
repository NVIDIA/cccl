// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/__functional/call_or.h>
#include <cuda/std/cstdint>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_segmented_exclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_segmented_exclusive_scan);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_segmented_inclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_segmented_inclusive_scan);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_segmented_inclusive_scan_init);

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_segmented_exclusive_sum_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_segmented_exclusive_scan_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_segmented_inclusive_sum_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_segmented_inclusive_scan_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit,
                       device_segmented_inclusive_scan_init_two_phase);

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
struct segmented_scan_tuning_transform
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability, cub::SegmentedScanPolicy policy) const
    -> cub::SegmentedScanPolicy
  {
    policy.block.threads_per_block = static_cast<int>(BlockThreads);
    return policy;
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

CUB_TEST("Device segmented inclusive scan tuning can be modified", "[segmented_scan][device]", CUB_SMALL)
{
  constexpr unsigned int target_block_size = 64; // must be different from any value we use in production
  ::cuda::std::int64_t num_segments        = 3;
  thrust::device_vector<int> d_offsets     = {0, 4, 7, 9};
  auto d_offsets_it                        = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size(), thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{thrust::raw_pointer_cast(d_block_size.data())};

  {
    device_segmented_inclusive_scan(
      d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, stdexec::env<>{});
    const unsigned int default_block_size = d_block_size[0];
    REQUIRE(default_block_size != target_block_size);
    d_block_size[0] = 0;
  }

  auto env = cuda::execution::tune(segmented_scan_tuning_transform<target_block_size>{});

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

#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH == 0

template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const stream_scope scope{cuda::__call_or(cuda::get_stream, cuda::stream_ref{cudaStream_t{}}, env).get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// The launch wrappers copy their arguments, which the non-copyable streams of test_with_custom_streams do not allow
CUB_TEST_CASE("Device segmented exclusive sum two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const c2h::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
          d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, env);
      });
      REQUIRE(d_out == expected);
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          env);
      });
      REQUIRE(d_out == expected);
    });
  }
}

CUB_TEST_CASE("Device segmented exclusive scan two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const c2h::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE(d_out == expected);
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE(d_out == expected);
    });
  }
}

CUB_TEST_CASE("Device segmented inclusive sum two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const c2h::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedSum(
          d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, env);
      });
      REQUIRE(d_out == expected);
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedSum(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          env);
      });
      REQUIRE(d_out == expected);
    });
  }
}

CUB_TEST_CASE("Device segmented inclusive scan two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const c2h::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          num_segments,
          cuda::std::plus<>{},
          env);
      });
      REQUIRE(d_out == expected);
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          cuda::std::plus<>{},
          env);
      });
      REQUIRE(d_out == expected);
    });
  }
}

CUB_TEST_CASE("Device segmented inclusive scan init two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const c2h::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE(d_out == expected);
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
          d_temp_storage,
          temp_storage_bytes,
          d_in.begin(),
          d_out.begin(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE(d_out == expected);
    });
  }
}

#endif // TEST_LAUNCH == 0

// The graph launch appends its own stream argument, which cannot be combined with an environment
#if TEST_LAUNCH != 2

CUB_TEST_CASE("Device segmented exclusive sum two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_exclusive_sum_two_phase(d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_exclusive_sum_two_phase(d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cudaStream_t{});
  }

  const c2h::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented exclusive scan two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }

  const c2h::device_vector<int> expected{100, 108, 114, 121, 100, 103, 103, 100, 101};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive sum two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_inclusive_sum_two_phase(d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_inclusive_sum_two_phase(d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cudaStream_t{});
  }

  const c2h::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive scan two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, cudaStream_t{});
  }

  const c2h::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device segmented inclusive scan init two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const ::cuda::std::int64_t num_segments = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  const c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  c2h::device_vector<int> d_out(d_in.size(), thrust::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.begin(), d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }

  const c2h::device_vector<int> expected{108, 114, 121, 126, 103, 103, 112, 101, 103};
  REQUIRE(d_out == expected);
}

// The tuning environments are empty classes, which older CTKs (at least 12.9 and 13.0) cannot pass as kernel
// parameters, so these tests cannot use the device launch (TEST_LAUNCH == 1).
#  if TEST_LAUNCH == 0

CUB_TEST("Device segmented exclusive sum two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_out(9, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  const auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_exclusive_sum_two_phase(d_in, d_out.begin(), offsets, offsets + 1, num_segments, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(d_in, d_out.begin(), offsets, offsets + 1, offsets, num_segments, env);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device segmented exclusive scan two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_out(9, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  const auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in, d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in, d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, env);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device segmented inclusive sum two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_out(9, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  const auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_inclusive_sum_two_phase(d_in, d_out.begin(), offsets, offsets + 1, num_segments, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(d_in, d_out.begin(), offsets, offsets + 1, offsets, num_segments, env);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device segmented inclusive scan two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_out(9, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  const auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in, d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in, d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, env);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "Device segmented inclusive scan init two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const c2h::device_vector<int> d_offsets{0, 4, 7, 9};
  const int* offsets = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_out(9, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const auto d_in = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  const auto env  = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in, d_out.begin(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in, d_out.begin(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, env);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

#  endif // TEST_LAUNCH == 0

#endif // TEST_LAUNCH != 2

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
#endif // _CCCL_COMPILER(GCC, >=, 8)

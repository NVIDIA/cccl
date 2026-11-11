// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

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
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/stream>

#include <cstddef>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>

namespace stdexec = cuda::std::execution;

// Test data (separate input/output offsets):
// Input data: {1,2,3,4,5,6,7,8}  - 3 segments of sizes 3, 2, 3 at input offsets {0,3,5,8}
// Output layout: 10 slots with padding at positions 3 and 6; output begin offsets {0,4,7}

using c2h::Equals;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device segmented exclusive sum works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
            d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 8, 14, 21, 0, 3, 3, 0, 1});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented exclusive scan works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
            d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {100, 108, 114, 121, 100, 103, 103, 100, 101});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive sum works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive scan works with default environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive scan init works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
            d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {108, 114, 121, 126, 103, 103, 112, 101, 103});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented exclusive sum with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, device, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, device, 10, sentinel);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
            d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented exclusive scan with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, device, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, device, 10, sentinel);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
      d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, 100));

  const auto expected =
    c2h::make_host_buffer<int>(stream, device, {100, 101, 103, sentinel, 100, 104, sentinel, 100, 106, 113});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive sum with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, device, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, device, 10, sentinel);

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive scan with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, device, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, device, 10, sentinel);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
      d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}));

  const auto expected = c2h::make_host_buffer<int>(stream, device, {1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive scan init with separate offsets works with default environment",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, device, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, device, 10, sentinel);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
      d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, 100));

  const auto expected =
    c2h::make_host_buffer<int>(stream, device, {101, 103, 106, sentinel, 104, 109, sentinel, 106, 113, 121});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

#endif // TEST_LAUNCH == 0

CUB_TEST("Device segmented exclusive sum uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
      nullptr, expected_bytes_allocated, d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_sum(d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {0, 8, 14, 21, 0, 3, 3, 0, 1});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented exclusive scan uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_offsets_it,
      d_offsets_it + 1,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_scan(
    d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {100, 108, 114, 121, 100, 103, 103, 100, 101});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented inclusive sum uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
      nullptr, expected_bytes_allocated, d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_sum(d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented inclusive scan uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_offsets_it,
      d_offsets_it + 1,
      num_segments,
      ::cuda::std::plus<>{}));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan(
    d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented inclusive scan init uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_offsets_it,
      d_offsets_it + 1,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan_init(
    d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 100, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {108, 114, 121, 126, 103, 103, 112, 101, 103});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented exclusive sum with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, 10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_sum(
    d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented exclusive scan with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, 10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_exclusive_scan(
    d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, 100, env);

  const auto expected =
    c2h::make_host_buffer<int>(stream, {100, 101, 103, sentinel, 100, 104, sentinel, 100, 106, 113});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented inclusive sum with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, 10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedSum(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_sum(
    d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented inclusive scan with separate offsets uses environment", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, 10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScan(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments,
      ::cuda::std::plus<>{}));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan(
    d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST("Device segmented inclusive scan init with separate offsets uses environment",
         "[segmented_scan][device]",
         CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  const auto sentinel      = -1;
  const auto num_segments  = ::cuda::std::int64_t{3};
  const auto d_in_offsets  = c2h::make_device_buffer<int>(stream, {0, 3, 5, 8});
  const auto d_out_offsets = c2h::make_device_buffer<int>(stream, {0, 4, 7});
  const auto d_in_off_it   = d_in_offsets.data();
  const auto d_out_off_it  = d_out_offsets.data();
  const auto d_in          = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out               = c2h::make_device_buffer<int>(stream, 10, sentinel);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      d_in_off_it,
      d_in_off_it + 1,
      d_out_off_it,
      num_segments,
      ::cuda::std::plus<>{},
      100));

  const auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_segmented_inclusive_scan_init(
    d_in.data(), d_out.data(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, ::cuda::std::plus<>{}, 100, env);

  const auto expected =
    c2h::make_host_buffer<int>(stream, {101, 103, 106, sentinel, 104, 109, sentinel, 106, 113, 121});
  REQUIRE_THAT(d_out, Equals(stream, expected));
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

  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  auto d_out              = c2h::make_device_buffer<int>(stream, 9, cuda::no_init);
  auto d_block_size       = c2h::make_device_buffer<unsigned int>(stream, {0u});
  auto d_in               = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};

  device_segmented_exclusive_sum(d_in, d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, env);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented inclusive sum can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  auto d_out              = c2h::make_device_buffer<int>(stream, 9, cuda::no_init);
  auto d_block_size       = c2h::make_device_buffer<unsigned int>(stream, {0u});
  auto d_in               = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};

  device_segmented_inclusive_sum(d_in, d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, env);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented exclusive scan can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);
  auto d_block_size       = c2h::make_device_buffer<unsigned int>(stream, {0u});
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{d_block_size.data()};
  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};

  device_segmented_exclusive_scan(
    d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {100, 108, 114, 121, 100, 103, 103, 100, 101});
  REQUIRE_THAT(d_out, Equals(stream, expected));
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented inclusive scan can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);
  auto d_block_size       = c2h::make_device_buffer<unsigned int>(stream, {0u});
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{d_block_size.data()};
  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};

  device_segmented_inclusive_scan(d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented inclusive scan tuning can be modified", "[segmented_scan][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr unsigned int target_block_size = 64; // must be different from any value we use in production
  const ::cuda::std::int64_t num_segments  = 3;

  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);
  auto d_block_size       = c2h::make_device_buffer<unsigned int>(stream, {0u});

  const block_size_extracting_op<::cuda::std::plus<>> scan_op{d_block_size.data()};

  {
    device_segmented_inclusive_scan(
      d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, cuda::stream_ref{stream});
    const unsigned int default_block_size = c2h::read_single(stream, d_block_size);
    REQUIRE(default_block_size != target_block_size);
  }

  cuda::fill_bytes(stream, d_block_size, 0);
  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(segmented_scan_tuning_transform<target_block_size>{})};

  device_segmented_inclusive_scan(d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented inclusive scan init can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto stream = c2h::make_current_device_stream();

  const auto num_segments = ::cuda::std::int64_t{3};
  const auto d_offsets    = c2h::make_device_buffer<int>(stream, {0, 4, 7, 9});
  const auto d_offsets_it = d_offsets.data();
  const auto d_in         = c2h::make_device_buffer<int>(stream, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = c2h::make_device_buffer<int>(stream, d_in.size(), cuda::no_init);
  auto d_block_size       = c2h::make_device_buffer<unsigned int>(stream, {0u});
  const block_size_extracting_op<::cuda::std::plus<>> scan_op{d_block_size.data()};
  const auto env =
    stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(segmented_scan_tuning<target_block_size>{})};

  device_segmented_inclusive_scan_init(
    d_in.data(), d_out.data(), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, 100, env);

  const auto expected = c2h::make_host_buffer<int>(stream, {108, 114, 121, 126, 103, 103, 112, 101, 103});
  REQUIRE_THAT(d_out, Equals(stream, expected));
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH == 0

template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const auto device = c2h::current_device();
  const auto stream = cuda::__call_or(cuda::get_stream, c2h::default_stream_ref(), env);
  const stream_scope scope{stream.get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, device, temp_storage_bytes, cuda::no_init);
  REQUIRE(cudaSuccess == two_phase(temp_storage.data(), temp_storage_bytes));
  stream.sync();
}

// The launch wrappers copy their arguments, which the non-copyable streams of test_with_custom_streams do not allow
CUB_TEST_CASE("Device segmented exclusive sum two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const auto expected                     = c2h::make_host_buffer<int>(stream, device, {0, 8, 14, 21, 0, 3, 3, 0, 1});

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
          d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }
}

CUB_TEST_CASE("Device segmented exclusive scan two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const auto expected = c2h::make_host_buffer<int>(stream, device, {100, 108, 114, 121, 100, 103, 103, 100, 101});

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }
}

CUB_TEST_CASE("Device segmented inclusive sum two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const auto expected                     = c2h::make_host_buffer<int>(stream, device, {8, 14, 21, 26, 3, 3, 12, 1, 3});

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedSum(
          d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedSum(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }
}

CUB_TEST_CASE("Device segmented inclusive scan two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const auto expected                     = c2h::make_host_buffer<int>(stream, device, {8, 14, 21, 26, 3, 3, 12, 1, 3});

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          num_segments,
          cuda::std::plus<>{},
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScan(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          cuda::std::plus<>{},
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }
}

CUB_TEST_CASE("Device segmented inclusive scan init two-phase works with custom streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const auto expected = c2h::make_host_buffer<int>(stream, device, {108, 114, 121, 126, 103, 103, 112, 101, 103});

  SECTION("common offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
    });
  }

  SECTION("separate output offsets")
  {
    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
          d_temp_storage,
          temp_storage_bytes,
          d_in.data(),
          d_out.data(),
          offsets,
          offsets + 1,
          offsets,
          num_segments,
          cuda::std::plus<>{},
          100,
          env);
      });
      REQUIRE_THAT(d_out, Equals(stream, expected));
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
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_exclusive_sum_two_phase(d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_exclusive_sum_two_phase(d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cudaStream_t{});
  }

  const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 8, 14, 21, 0, 3, 3, 0, 1});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented exclusive scan two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }

  const auto expected = c2h::make_host_buffer<int>(stream, device, {100, 108, 114, 121, 100, 103, 103, 100, 101});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive sum two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_inclusive_sum_two_phase(d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_inclusive_sum_two_phase(d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cudaStream_t{});
  }

  const auto expected = c2h::make_host_buffer<int>(stream, device, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive scan two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, cudaStream_t{});
  }

  const auto expected = c2h::make_host_buffer<int>(stream, device, {8, 14, 21, 26, 3, 3, 12, 1, 3});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

CUB_TEST_CASE("Device segmented inclusive scan init two-phase accepts legacy null streams",
              "[segmented_scan][device]",
              CUB_SMALL)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  const ::cuda::std::int64_t num_segments = 3;
  const auto d_offsets                    = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                      = d_offsets.data();
  const auto d_in                         = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out                              = c2h::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }
  SECTION("nullptr with separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, nullptr);
  }
  SECTION("literal 0 with separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, 0);
  }
  SECTION("cudaStream_t{} with separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in.data(), d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, cudaStream_t{});
  }

  const auto expected = c2h::make_host_buffer<int>(stream, device, {108, 114, 121, 126, 103, 103, 112, 101, 103});
  REQUIRE_THAT(d_out, Equals(stream, expected));
}

// The tuning environments are empty classes, which older CTKs (at least 12.9 and 13.0) cannot pass as kernel
// parameters, so these tests cannot use the device launch (TEST_LAUNCH == 1).
#  if TEST_LAUNCH == 0

CUB_TEST("Device segmented exclusive sum two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const auto d_offsets                     = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                       = d_offsets.data();
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 9, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env                           = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_exclusive_sum_two_phase(d_in, d_out.data(), offsets, offsets + 1, num_segments, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_exclusive_sum_two_phase(d_in, d_out.data(), offsets, offsets + 1, offsets, num_segments, env);
  }
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented exclusive scan two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const auto d_offsets                     = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                       = d_offsets.data();
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 9, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env                           = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in, d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_exclusive_scan_two_phase(
      d_in, d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, env);
  }
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented inclusive sum two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const auto d_offsets                     = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                       = d_offsets.data();
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 9, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env                           = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_inclusive_sum_two_phase(d_in, d_out.data(), offsets, offsets + 1, num_segments, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_inclusive_sum_two_phase(d_in, d_out.data(), offsets, offsets + 1, offsets, num_segments, env);
  }
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device segmented inclusive scan two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const auto d_offsets                     = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                       = d_offsets.data();
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 9, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env                           = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in, d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_inclusive_scan_two_phase(
      d_in, d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, env);
  }
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST(
  "Device segmented inclusive scan init two-phase can be tuned", "[segmented_scan][device]", CUB_SMALL, block_sizes)
{
  const auto device = c2h::current_device();
  const auto stream = c2h::default_stream_ref();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const ::cuda::std::int64_t num_segments  = 3;
  const auto d_offsets                     = c2h::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  const int* offsets                       = d_offsets.data();
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 9, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  const auto env                           = cuda::execution::tune(segmented_scan_tuning<target_block_size>{});

  SECTION("common offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in, d_out.data(), offsets, offsets + 1, num_segments, cuda::std::plus<>{}, 100, env);
  }
  SECTION("separate output offsets")
  {
    device_segmented_inclusive_scan_init_two_phase(
      d_in, d_out.data(), offsets, offsets + 1, offsets, num_segments, cuda::std::plus<>{}, 100, env);
  }
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
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

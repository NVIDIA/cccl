// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/stream>

#include <cstddef>
#include <iostream>
#include <vector>

#include "catch2_test_device_segmented_scan_utils.cuh"
#include "cub_test_macros.h"

using segmented_scan_test::current_device;
using segmented_scan_test::require_equal;

CUB_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum accepts stream", "[segmented_scan][env]", CUB_SMALL)
{
  // example-begin exclusive-segmented-sum-env
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_offsets          = cuda::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  auto d_offsets_it       = d_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << error << '\n';
  }

  const std::vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  // example-end exclusive-segmented-sum-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedScan accepts stream", "[segmented_scan][env]", CUB_SMALL)
{
  // example-begin exclusive-segmented-scan-env
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_offsets          = cuda::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  auto d_offsets_it       = d_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, cuda::maximum<>{}, 4, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedScan failed with status: " << error << '\n';
  }

  const std::vector<int> expected{4, 8, 8, 8, 4, 4, 4, 4, 4};
  // example-end exclusive-segmented-scan-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum accepts stream", "[segmented_scan][env]", CUB_SMALL)
{
  // example-begin inclusive-segmented-sum-env
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_offsets          = cuda::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  auto d_offsets_it       = d_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << error << '\n';
  }

  const std::vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  // example-end inclusive-segmented-sum-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScan accepts stream", "[segmented_scan][env]", CUB_SMALL)
{
  // example-begin inclusive-segmented-scan-env
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_offsets          = cuda::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  auto d_offsets_it       = d_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, cuda::maximum<>{}, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScan failed with status: " << error << '\n';
  }

  const std::vector<int> expected{8, 8, 8, 8, 3, 3, 9, 1, 2};
  // example-end inclusive-segmented-scan-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit accepts stream", "[segmented_scan][env]", CUB_SMALL)
{
  // example-begin inclusive-segmented-scan-init-env
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_offsets          = cuda::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  auto d_offsets_it       = d_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, cuda::maximum<>{}, 4, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit failed with status: " << error << '\n';
  }

  const std::vector<int> expected{8, 8, 8, 8, 4, 4, 9, 4, 4};
  // example-end inclusive-segmented-scan-init-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum (separate offsets) accepts stream",
         "[segmented_scan][env]",
         CUB_SMALL)
{
  // example-begin exclusive-segmented-sum-separate-env
  const auto sentinel     = -1;
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_in_offsets       = cuda::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  auto d_out_offsets      = cuda::make_device_buffer<int>(stream, device, {0, 4, 7});
  auto d_in_off_it        = d_in_offsets.begin();
  auto d_out_off_it       = d_out_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, 10, sentinel);

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << error << '\n';
  }

  const std::vector<int> expected{0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13};
  // example-end exclusive-segmented-sum-separate-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedScan (separate offsets) accepts stream",
         "[segmented_scan][env]",
         CUB_SMALL)
{
  // example-begin exclusive-segmented-scan-separate-env
  const auto sentinel     = -1;
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_in_offsets       = cuda::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  auto d_out_offsets      = cuda::make_device_buffer<int>(stream, device, {0, 4, 7});
  auto d_in_off_it        = d_in_offsets.begin();
  auto d_out_off_it       = d_out_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {3, 1, 4, 1, 5, 9, 2, 6});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, 10, sentinel);

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, cuda::maximum<>{}, 2, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedScan failed with status: " << error << '\n';
  }

  const std::vector<int> expected{2, 3, 3, sentinel, 2, 2, sentinel, 2, 9, 9};
  // example-end exclusive-segmented-scan-separate-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum (separate offsets) accepts stream",
         "[segmented_scan][env]",
         CUB_SMALL)
{
  // example-begin inclusive-segmented-sum-separate-env
  const auto sentinel     = -1;
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_in_offsets       = cuda::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  auto d_out_offsets      = cuda::make_device_buffer<int>(stream, device, {0, 4, 7});
  auto d_in_off_it        = d_in_offsets.begin();
  auto d_out_off_it       = d_out_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {1, 2, 3, 4, 5, 6, 7, 8});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, 10, sentinel);

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << error << '\n';
  }

  const std::vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  // example-end inclusive-segmented-sum-separate-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScan (separate offsets) accepts stream",
         "[segmented_scan][env]",
         CUB_SMALL)
{
  // example-begin inclusive-segmented-scan-separate-env
  const auto sentinel     = -1;
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_in_offsets       = cuda::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  auto d_out_offsets      = cuda::make_device_buffer<int>(stream, device, {0, 4, 7});
  auto d_in_off_it        = d_in_offsets.begin();
  auto d_out_off_it       = d_out_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {3, 1, 4, 1, 5, 9, 2, 6});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, 10, sentinel);

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, cuda::maximum<>{}, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScan failed with status: " << error << '\n';
  }

  const std::vector<int> expected{3, 3, 4, sentinel, 1, 5, sentinel, 9, 9, 9};
  // example-end inclusive-segmented-scan-separate-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

CUB_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit (separate offsets) accepts stream",
         "[segmented_scan][env]",
         CUB_SMALL)
{
  // example-begin inclusive-segmented-scan-init-separate-env
  const auto sentinel     = -1;
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto d_in_offsets       = cuda::make_device_buffer<int>(stream, device, {0, 3, 5, 8});
  auto d_out_offsets      = cuda::make_device_buffer<int>(stream, device, {0, 4, 7});
  auto d_in_off_it        = d_in_offsets.begin();
  auto d_out_off_it       = d_out_offsets.begin();
  auto d_in               = cuda::make_device_buffer<int>(stream, device, {3, 1, 4, 1, 5, 9, 2, 6});
  auto d_out              = cuda::make_device_buffer<int>(stream, device, 10, sentinel);

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, cuda::maximum<>{}, 7, stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit failed with status: " << error << '\n';
  }

  const std::vector<int> expected{7, 7, 7, sentinel, 7, 7, sentinel, 9, 9, 9};
  // example-end inclusive-segmented-scan-init-separate-env

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

#if _CCCL_STD_VER >= 2020

// example-begin segmented-scan-policy-selector
struct SegmentedScanPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::SegmentedScanPolicy
  {
    return {.block = {.threads_per_block = 128,
                      .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 11 : 9,
                      .load_algorithm    = cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      .load_modifier     = cub::LOAD_DEFAULT,
                      .store_algorithm   = cub::BLOCK_STORE_WARP_TRANSPOSE,
                      .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
                      .max_segments      = 512}};
  }
};
// example-end segmented-scan-policy-selector

CUB_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedScan accepts a custom policy selector",
         "[segmented_scan][env]",
         CUB_SMALL)
{
  // example-begin segmented-scan-tuning
  const auto num_segments = ::cuda::std::int64_t{3};
  auto device             = current_device();
  auto stream             = cuda::stream{device};
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, cuda::execution::tune(SegmentedScanPolicySelector{})};
  auto d_offsets    = cuda::make_device_buffer<int>(stream, device, {0, 4, 7, 9});
  auto d_offsets_it = d_offsets.begin();
  auto d_in         = cuda::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9, 1, 2});
  auto d_out        = cuda::make_device_buffer<int>(stream, device, d_in.size(), cuda::no_init);

  const auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, ::cuda::std::plus<>{}, 0, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedScan failed with status: " << error << '\n';
  }

  const std::vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  // example-end segmented-scan-tuning

  REQUIRE(error == cudaSuccess);
  require_equal(stream, d_out, expected);
}

#endif // _CCCL_STD_VER >= 2020

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/functional>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum accepts stream", "[segmented_scan][env]")
{
  // example-begin exclusive-segmented-sum-env
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{0, 8, 14, 21, 0, 3, 3, 0, 1};
  // example-end exclusive-segmented-sum-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedScan accepts stream", "[segmented_scan][env]")
{
  // example-begin exclusive-segmented-scan-env
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, cuda::maximum<>{}, 4, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedScan failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{4, 8, 8, 8, 4, 4, 4, 4, 4};
  // example-end exclusive-segmented-scan-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum accepts stream", "[segmented_scan][env]")
{
  // example-begin inclusive-segmented-sum-env
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 14, 21, 26, 3, 3, 12, 1, 3};
  // example-end inclusive-segmented-sum-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScan accepts stream", "[segmented_scan][env]")
{
  // example-begin inclusive-segmented-scan-env
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, cuda::maximum<>{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScan failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 8, 8, 8, 3, 3, 9, 1, 2};
  // example-end inclusive-segmented-scan-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit accepts stream", "[segmented_scan][env]")
{
  // example-begin inclusive-segmented-scan-init-env
  ::cuda::std::int64_t num_segments    = 3;
  thrust::device_vector<int> d_offsets = {0, 4, 7, 9};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9, 1, 2};
  thrust::device_vector<int> d_out(d_in.size());

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    d_in.begin(), d_out.begin(), d_offsets_it, d_offsets_it + 1, num_segments, cuda::maximum<>{}, 4, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 8, 8, 8, 4, 4, 9, 4, 4};
  // example-end inclusive-segmented-scan-init-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedSum (separate offsets) accepts stream", "[segmented_scan][env]")
{
  // example-begin exclusive-segmented-sum-separate-env
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedSum failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{0, 1, 3, sentinel, 0, 4, sentinel, 0, 6, 13};
  // example-end exclusive-segmented-sum-separate-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::ExclusiveSegmentedScan (separate offsets) accepts stream", "[segmented_scan][env]")
{
  // example-begin exclusive-segmented-scan-separate-env
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{3, 1, 4, 1, 5, 9, 2, 6};
  thrust::device_vector<int> d_out(10, sentinel);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
    d_in.begin(),
    d_out.begin(),
    d_in_off_it,
    d_in_off_it + 1,
    d_out_off_it,
    num_segments,
    cuda::maximum<>{},
    2,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::ExclusiveSegmentedScan failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{2, 3, 3, sentinel, 2, 2, sentinel, 2, 9, 9};
  // example-end exclusive-segmented-scan-separate-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedSum (separate offsets) accepts stream", "[segmented_scan][env]")
{
  // example-begin inclusive-segmented-sum-separate-env
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(10, sentinel);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedSum(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedSum failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{1, 3, 6, sentinel, 4, 9, sentinel, 6, 13, 21};
  // example-end inclusive-segmented-sum-separate-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScan (separate offsets) accepts stream", "[segmented_scan][env]")
{
  // example-begin inclusive-segmented-scan-separate-env
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{3, 1, 4, 1, 5, 9, 2, 6};
  thrust::device_vector<int> d_out(10, sentinel);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScan(
    d_in.begin(), d_out.begin(), d_in_off_it, d_in_off_it + 1, d_out_off_it, num_segments, cuda::maximum<>{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScan failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{3, 3, 4, sentinel, 1, 5, sentinel, 9, 9, 9};
  // example-end inclusive-segmented-scan-separate-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit (separate offsets) accepts stream",
         "[segmented_scan][env]")
{
  // example-begin inclusive-segmented-scan-init-separate-env
  const auto sentinel               = -1;
  ::cuda::std::int64_t num_segments = 3;
  thrust::device_vector<int> d_in_offsets{0, 3, 5, 8};
  thrust::device_vector<int> d_out_offsets{0, 4, 7};
  auto d_in_off_it  = thrust::raw_pointer_cast(d_in_offsets.data());
  auto d_out_off_it = thrust::raw_pointer_cast(d_out_offsets.data());
  thrust::device_vector<int> d_in{3, 1, 4, 1, 5, 9, 2, 6};
  thrust::device_vector<int> d_out(10, sentinel);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedScan::InclusiveSegmentedScanInit(
    d_in.begin(),
    d_out.begin(),
    d_in_off_it,
    d_in_off_it + 1,
    d_out_off_it,
    num_segments,
    cuda::maximum<>{},
    7,
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{7, 7, 7, sentinel, 7, 7, sentinel, 9, 9, 9};
  // example-end inclusive-segmented-scan-init-separate-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out == expected);
}

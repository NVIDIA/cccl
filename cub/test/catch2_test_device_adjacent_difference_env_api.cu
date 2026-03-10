// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceAdjacentDifference::SubtractLeftCopy accepts stream", "[adjacent_difference][env]")
{
  // example-begin subtract-left-copy-env-stream
  auto input  = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = thrust::device_vector<int>(8);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceAdjacentDifference::SubtractLeftCopy(
    input.begin(), output.begin(), input.size(), cuda::std::minus{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractLeftCopy failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  // example-end subtract-left-copy-env-stream
  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceAdjacentDifference::SubtractLeft accepts stream", "[adjacent_difference][env]")
{
  // example-begin subtract-left-env-stream
  auto data = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceAdjacentDifference::SubtractLeft(data.begin(), data.size(), cuda::std::minus{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractLeft failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{1, 1, -1, 1, -1, 1, -1, 1};
  // example-end subtract-left-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(data == expected);
}

C2H_TEST("cub::DeviceAdjacentDifference::SubtractRightCopy accepts stream", "[adjacent_difference][env]")
{
  // example-begin subtract-right-copy-env-stream
  auto input  = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};
  auto output = thrust::device_vector<int>(8);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceAdjacentDifference::SubtractRightCopy(
    input.begin(), output.begin(), input.size(), cuda::std::minus{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractRightCopy failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  // example-end subtract-right-copy-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceAdjacentDifference::SubtractRight accepts stream", "[adjacent_difference][env]")
{
  // example-begin subtract-right-env-stream
  auto data = thrust::device_vector<int>{1, 2, 1, 2, 1, 2, 1, 2};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceAdjacentDifference::SubtractRight(data.begin(), data.size(), cuda::std::minus{}, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceAdjacentDifference::SubtractRight failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{-1, 1, -1, 1, -1, 1, -1, 2};
  // example-end subtract-right-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(data == expected);
}

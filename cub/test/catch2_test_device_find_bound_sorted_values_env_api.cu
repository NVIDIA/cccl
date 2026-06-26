// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceFind::LowerBoundSortedValues accepts env with stream", "[find][env][binary-search]")
{
  // example-begin lower-bound-sorted-values-env
  thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_values = {0, 3, 4, 7};
  thrust::device_vector<int> d_output(4);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceFind::LowerBoundSortedValues(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::LowerBoundSortedValues failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected = {0, 2, 2, 4};
  // example-end lower-bound-sorted-values-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

C2H_TEST("cub::DeviceFind::UpperBoundSortedValues accepts env with stream", "[find][env][binary-search]")
{
  // example-begin upper-bound-sorted-values-env
  thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_values = {0, 3, 4, 7};
  thrust::device_vector<int> d_output(4);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceFind::UpperBoundSortedValues(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::UpperBoundSortedValues failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected = {1, 2, 3, 4};
  // example-end upper-bound-sorted-values-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

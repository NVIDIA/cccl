// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

// example-begin find-if-env-predicate
struct is_greater_than_t
{
  int threshold;
  __host__ __device__ bool operator()(int value) const
  {
    return value > threshold;
  }
};
// example-end find-if-env-predicate

C2H_TEST("cub::DeviceFind::FindIf accepts env with stream", "[find][env]")
{
  // example-begin find-if-env
  constexpr int num_items         = 8;
  thrust::device_vector<int> d_in = {0, 1, 2, 3, 4, 5, 6, 7};
  thrust::device_vector<int> d_out(1);
  is_greater_than_t predicate{4};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceFind::FindIf(d_in.begin(), d_out.begin(), predicate, num_items, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::FindIf failed with status: " << error << '\n';
  }

  int expected = 5;
  // example-end find-if-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out[0] == expected);
}

C2H_TEST("cub::DeviceFind::LowerBound accepts env with stream", "[find][env]")
{
  // example-begin lower-bound-env
  thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_values = {1, 3, 5, 7};
  thrust::device_vector<int> d_output(4);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceFind::LowerBound(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::LowerBound failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected = {1, 2, 3, 4};
  // example-end lower-bound-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

C2H_TEST("cub::DeviceFind::UpperBound accepts env with stream", "[find][env]")
{
  // example-begin upper-bound-env
  thrust::device_vector<int> d_range  = {0, 2, 4, 6, 8};
  thrust::device_vector<int> d_values = {1, 3, 5, 7};
  thrust::device_vector<int> d_output(4);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceFind::UpperBound(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceFind::UpperBound failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected = {1, 2, 3, 4};
  // example-end upper-bound-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

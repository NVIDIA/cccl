// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include "cub_test_macros.h"

CUB_TEST("cub::DeviceReduce::ArgMinMax API example", "[reduce][env]", CUB_SMALL)
{
  // example-begin argminmax-env
  auto input      = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output = thrust::device_vector<float>(1, thrust::no_init);
  auto min_index  = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_output = thrust::device_vector<float>(1, thrust::no_init);
  auto max_index  = thrust::device_vector<cuda::std::int64_t>(1, thrust::no_init);

  cuda::stream stream{cuda::devices[0]};

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_output.begin(),
    min_index.begin(),
    max_output.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMinMax failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_min_index{3};
  thrust::device_vector<float> expected_max{4.0f};
  thrust::device_vector<cuda::std::int64_t> expected_max_index{2};
  // example-end argminmax-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(min_index == expected_min_index);
  REQUIRE(max_output == expected_max);
  REQUIRE(max_index == expected_max_index);
}

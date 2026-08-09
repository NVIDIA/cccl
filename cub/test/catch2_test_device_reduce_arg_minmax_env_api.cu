// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/std/__execution/env.h>
#include <cuda/stream>

#include "cub_test_macros.h"

CUB_TEST("cub::DeviceReduce::ArgMinMax accepts determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin argminmax-env-determinism
  auto input      = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output = thrust::device_vector<float>(1);
  auto min_index  = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_output = thrust::device_vector<float>(1);
  auto max_index  = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_output.begin(),
    min_index.begin(),
    max_output.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMinMax failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_min_index{3};
  thrust::device_vector<float> expected_max{4.0f};
  thrust::device_vector<cuda::std::int64_t> expected_max_index{2};
  // example-end argminmax-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(min_index == expected_min_index);
  REQUIRE(max_output == expected_max);
  REQUIRE(max_index == expected_max_index);
}

CUB_TEST("cub::DeviceReduce::ArgMinMax accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin argminmax-env-stream
  auto input      = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output = thrust::device_vector<float>(1);
  auto min_index  = thrust::device_vector<cuda::std::int64_t>(1);
  auto max_output = thrust::device_vector<float>(1);
  auto max_index  = thrust::device_vector<cuda::std::int64_t>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ArgMinMax(
    input.begin(),
    min_output.begin(),
    min_index.begin(),
    max_output.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMinMax failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_min_index{3};
  thrust::device_vector<float> expected_max{4.0f};
  thrust::device_vector<cuda::std::int64_t> expected_max_index{2};
  // example-end argminmax-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(min_index == expected_min_index);
  REQUIRE(max_output == expected_max);
  REQUIRE(max_index == expected_max_index);
}

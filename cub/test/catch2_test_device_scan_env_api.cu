// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceScan::ExclusiveScan accepts run_to_run determinism requirements", "[scan][env]")
{
  // example-begin exclusive-scan-env-determinism
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<int>{0, 1, 2, 3};
  auto output = thrust::device_vector<int>(4);
  auto init   = 0;

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceScan::ExclusiveScan(input.begin(), output.begin(), op, init, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{0, 0, 1, 3};
  // example-end exclusive-scan-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan accepts stream and not_guaranteed determinism", "[scan][env]")
{
  // example-begin exclusive-scan-env-stream
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(4);
  auto init   = 1.0f;

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto req_env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  auto env     = cuda::std::execution::env{stream_ref, req_env};

  auto error = cub::DeviceScan::ExclusiveScan(input.begin(), output.begin(), op, init, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{1.0f, 1.0f, 2.0f, 4.0f};
  // example-end exclusive-scan-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::ExclusiveSum accepts run_to_run determinism requirements", "[scan][env]")
{
  // example-begin exclusive-sum-env-determinism
  auto input  = thrust::device_vector<int>{0, 1, 2, 3};
  auto output = thrust::device_vector<int>(4);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceScan::ExclusiveSum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveSum failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{0, 0, 1, 3};
  // example-end exclusive-sum-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::ExclusiveSum accepts stream and not_guaranteed determinism", "[scan][env]")
{
  // example-begin exclusive-sum-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(4);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto req_env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  auto env     = cuda::std::execution::env{stream_ref, req_env};

  auto error = cub::DeviceScan::ExclusiveSum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveSum failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{0.0f, 0.0f, 1.0f, 3.0f};
  // example-end exclusive-sum-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

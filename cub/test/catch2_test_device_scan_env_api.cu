// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/devices>
#include <cuda/stream>

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
    std::cerr << "cub::DeviceScan::ExclusiveScan failed with status: " << error << '\n';
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

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto req_env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  auto env     = cuda::std::execution::env{stream_ref, req_env};

  auto error = cub::DeviceScan::ExclusiveScan(input.begin(), output.begin(), op, init, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{1.0f, 1.0f, 2.0f, 4.0f};
  // example-end exclusive-scan-env-stream
  stream.sync();

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
    std::cerr << "cub::DeviceScan::ExclusiveSum failed with status: " << error << '\n';
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

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto req_env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  auto env     = cuda::std::execution::env{stream_ref, req_env};

  auto error = cub::DeviceScan::ExclusiveSum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveSum failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{0.0f, 0.0f, 1.0f, 3.0f};
  // example-end exclusive-sum-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveSum accepts run_to_run determinism requirements", "[scan][env]")
{
  // example-begin inclusive-sum-env-determinism
  auto input  = thrust::device_vector<int>{1, 2, 3, 4};
  auto output = thrust::device_vector<int>(4);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceScan::InclusiveSum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveSum failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{1, 3, 6, 10};
  // example-end inclusive-sum-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveSum accepts stream and not_guaranteed determinism", "[scan][env]")
{
  // example-begin inclusive-sum-env-stream
  auto input  = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output = thrust::device_vector<float>(4);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto req_env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  auto env     = cuda::std::execution::env{stream_ref, req_env};

  auto error = cub::DeviceScan::InclusiveSum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveSum failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{1.0f, 3.0f, 6.0f, 10.0f};
  // example-end inclusive-sum-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan with FutureValue accepts environment", "[scan][env]")
{
  // example-begin exclusive-scan-future-env
  auto input  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto output = thrust::device_vector<int>(7);

  auto init_value_vec = thrust::device_vector<int>{5};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceScan::ExclusiveScan(input.begin(), output.begin(), cuda::std::plus{}, future_init, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan (FutureValue) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{5, 13, 19, 26, 31, 34, 34};
  // example-end exclusive-scan-future-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::ExclusiveScan with FutureValue accepts stream environment", "[scan][env]")
{
  // example-begin exclusive-scan-future-env-stream
  auto input  = thrust::device_vector<int>{1, 2, 3, 4};
  auto output = thrust::device_vector<int>(4);

  auto init_value_vec = thrust::device_vector<int>{10};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceScan::ExclusiveScan(input.begin(), output.begin(), cuda::std::plus{}, future_init, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan (FutureValue) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{10, 11, 13, 16};
  // example-end exclusive-scan-future-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveScan accepts environment", "[scan][env]")
{
  // example-begin inclusive-scan-env
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<int>{1, 2, 3, 4};
  auto output = thrust::device_vector<int>(4);

  auto error = cub::DeviceScan::InclusiveScan(input.begin(), output.begin(), op, input.size());
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveScan failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{1, 3, 6, 10};
  // example-end inclusive-scan-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveScan accepts stream environment", "[scan][env]")
{
  // example-begin inclusive-scan-env-stream
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output = thrust::device_vector<float>(4);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceScan::InclusiveScan(input.begin(), output.begin(), op, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveScan failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{1.0f, 3.0f, 6.0f, 10.0f};
  // example-end inclusive-scan-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveScanInit accepts environment", "[scan][env]")
{
  // example-begin inclusive-scan-init-env
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<int>{1, 2, 3, 4};
  auto output = thrust::device_vector<int>(4);
  auto init   = 10;

  auto error = cub::DeviceScan::InclusiveScanInit(input.begin(), output.begin(), op, init, input.size());
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveScanInit failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{11, 13, 16, 20};
  // example-end inclusive-scan-init-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceScan::InclusiveScanInit accepts stream environment", "[scan][env]")
{
  // example-begin inclusive-scan-init-env-stream
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output = thrust::device_vector<float>(4);
  auto init   = 10.0f;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceScan::InclusiveScanInit(input.begin(), output.begin(), op, init, input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::InclusiveScanInit failed with status: " << error << '\n';
  }

  thrust::device_vector<float> expected{11.0f, 13.0f, 16.0f, 20.0f};
  // example-end inclusive-scan-init-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

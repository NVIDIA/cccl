// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceReduce::Reduce accepts determinism requirements", "[reduce][env]")
{
  // TODO(srinivas): replace with gpu_to_gpu once offset size restriction is relaxed
  // example-begin reduce-env-determinism
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);
  auto init   = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  // example-end reduce-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Reduce accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin reduce-env-non-determinism
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);
  auto init   = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  // example-end reduce-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Reduce accepts stream", "[reduce][env]")
{
  // example-begin reduce-env-stream
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);
  auto init   = 0.0f;

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  // example-end reduce-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts stream", "[reduce][env]")
{
  // example-begin sum-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    thrust::device_vector<float> expected{6.0f};
  }
  // example-end sum-env-stream
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceReduce::Sum accepts determinism requirements", "[reduce][env]")
{
  // TODO(srinivas): replace with gpu_to_gpu once offset size restriction is relaxed
  // example-begin sum-env-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  // example-end sum-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin sum-env-non-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  // example-end sum-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts stream", "[reduce][env]")
{
  // example-begin sum-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  // example-end sum-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Min accepts determinism requirements", "[reduce][env]")
{
  // example-begin min-env-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{0.0f};
  // example-end min-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Min accepts stream", "[reduce][env]")
{
  // example-begin min-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{0.0f};
  // example-end min-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Max accepts determinism requirements", "[reduce][env]")
{
  // example-begin max-env-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{3.0f};
  // example-end max-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Max accepts stream", "[reduce][env]")
{
  // example-begin max-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{3.0f};
  // example-end max-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::ArgMin accepts determinism requirements", "[reduce][env]")
{
  // example-begin argmin-env-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMin accepts stream", "[reduce][env]")
{
  // example-begin argmin-env-stream
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error =
    cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMax accepts determinism requirements", "[reduce][env]")
{
  // example-begin argmax-env-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ArgMax(input.begin(), max_output.begin(), index_output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMax failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected_max{4.0f};
  thrust::device_vector<cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output == expected_max);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMax accepts stream", "[reduce][env]")
{
  // example-begin argmax-env-stream
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error =
    cub::DeviceReduce::ArgMax(input.begin(), max_output.begin(), index_output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMax failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected_max{4.0f};
  thrust::device_vector<cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output == expected_max);
  REQUIRE(index_output == expected_index);
}

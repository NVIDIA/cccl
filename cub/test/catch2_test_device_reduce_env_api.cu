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
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin reduce-env-determinism
  auto op     = cuda::std::plus{};
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);
  auto init   = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, env);

  c2h::device_vector<float> expected{6.0f};
  // example-end reduce-env-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Reduce accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin reduce-env-non-determinism
  auto op     = cuda::std::plus{};
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);
  auto init   = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, env);

  c2h::device_vector<float> expected{6.0f};
  // example-end reduce-env-non-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Reduce accepts stream", "[reduce][env]")
{
  // example-begin reduce-env-stream
  auto op     = cuda::std::plus{};
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);
  auto init   = 0.0f;

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, stream_ref);

  c2h::device_vector<float> expected{6.0f};
  // example-end reduce-env-stream

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts stream", "[reduce][env]")
{
  // example-begin sum-env-stream
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), stream_ref);

  c2h::device_vector<float> expected{6.0f};
  // example-end sum-env-stream
}

C2H_TEST("cub::DeviceReduce::Min accepts determinism requirements", "[reduce][env]")
{
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin min-env-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{0.0f};
  // example-end min-env-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin sum-env-non-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{6.0f};
  // example-end sum-env-non-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts stream", "[reduce][env]")
{
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin max-env-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{3.0f};
  // example-end max-env-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::ArgMin accepts determinism requirements", "[reduce][env]")
{
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin argmin-env-determinism
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<::cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), input.size(), env);

  c2h::device_vector<float> expected_min{0.0f};
  c2h::device_vector<::cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-determinism

  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMax accepts determinism requirements", "[reduce][env]")
{
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin argmin-env-determinism
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<::cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::ArgMax(input.begin(), min_output.begin(), index_output.begin(), input.size(), env);

  c2h::device_vector<float> expected_min{4.0f};
  c2h::device_vector<::cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-determinism

  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

template <class T>
struct square_t
{
  __host__ __device__ T operator()(const T& x) const
  {
    return x * x;
  }
};

C2H_TEST("cub::DeviceReduce::TransformReduce accepts determinism requirements", "[reduce][env]")
{
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin transform-reduce-env-determinism
  auto input  = c2h::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::TransformReduce(
    input.begin(), output.begin(), input.size(), ::cuda::std::plus<float>{}, square_t<float>{}, 0.0f, env);

  c2h::device_vector<float> expected{30.0f};
  // example-end transform-reduce-env-determinism

  REQUIRE(output == expected);
}

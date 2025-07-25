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

C2H_TEST("cub::DeviceReduce::Sum accepts determinism requirements", "[reduce][env]")
{
  // TODO(gevtushenko): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin sum-env-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{6.0f};
  // example-end sum-env-determinism

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
  // example-begin sum-env-stream
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), stream_ref);

  c2h::device_vector<float> expected{6.0f};
  // example-end sum-env-stream

  REQUIRE(output == expected);
}

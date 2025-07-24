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
  // TODO(gonidelis): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

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
  // TODO(gonidelis): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

  // example-begin sum-env-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{6.0f};
  // example-end sum-env-determinism

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
  // example-begin min-env-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{0.0f};
  // example-end min-env-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Min accepts stream", "[reduce][env]")
{
  // example-begin min-env-stream
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), stream_ref);

  c2h::device_vector<float> expected{0.0f};
  // example-end min-env-stream

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Max accepts determinism requirements", "[reduce][env]")
{
  // example-begin max-env-determinism
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), env);

  c2h::device_vector<float> expected{3.0f};
  // example-end max-env-determinism

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Max accepts stream", "[reduce][env]")
{
  // example-begin max-env-stream
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), stream_ref);

  c2h::device_vector<float> expected{3.0f};
  // example-end max-env-stream

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::ArgMin accepts determinism requirements", "[reduce][env]")
{
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

C2H_TEST("cub::DeviceReduce::ArgMin accepts stream", "[reduce][env]")
{
  // example-begin argmin-env-stream
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<::cuda::std::int64_t>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), input.size(), stream_ref);

  c2h::device_vector<float> expected_min{0.0f};
  c2h::device_vector<::cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-stream

  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMax accepts determinism requirements", "[reduce][env]")
{
  // example-begin argmax-env-determinism
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<::cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  cub::DeviceReduce::ArgMax(input.begin(), min_output.begin(), index_output.begin(), input.size(), env);

  c2h::device_vector<float> expected_min{4.0f};
  c2h::device_vector<::cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-determinism

  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMax accepts stream", "[reduce][env]")
{
  // example-begin argmax-env-stream
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<::cuda::std::int64_t>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::ArgMax(input.begin(), min_output.begin(), index_output.begin(), input.size(), stream_ref);

  c2h::device_vector<float> expected_min{4.0f};
  c2h::device_vector<::cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-stream

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
  // TODO(gonidelis): replace `run_to_run` with `gpu_to_gpu` once RFA unwraps contiguous iterators

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

C2H_TEST("cub::DeviceReduce::TransformReduce accepts stream", "[reduce][env]")
{
  // example-begin transform-reduce-env-stream
  auto input  = c2h::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = c2h::device_vector<float>(1);
  auto init   = 0.0f;

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  cub::DeviceReduce::TransformReduce(
    input.begin(), output.begin(), input.size(), ::cuda::std::plus<float>{}, square_t<float>{}, init, stream_ref);

  c2h::device_vector<float> expected{14.0f};
  // example-end transform-reduce-env-stream

  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::ReduceByKey accepts determinism requirements", "[reduce][env]")
{
  // example-begin reduce-by-key-env-determinism
  auto keys_in     = c2h::device_vector<int>{1, 1, 2, 2, 2, 3, 3};
  auto values_in   = c2h::device_vector<float>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f};
  auto unique_keys = c2h::device_vector<int>(3);
  auto aggregates  = c2h::device_vector<float>(3);
  auto num_runs    = c2h::device_vector<int>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys.begin(),
    values_in.begin(),
    aggregates.begin(),
    num_runs.begin(),
    ::cuda::std::plus<float>{},
    keys_in.size(),
    env);

  c2h::device_vector<int> expected_keys{1, 2, 3};
  c2h::device_vector<float> expected_aggregates{30.0f, 120.0f, 130.0f};
  c2h::device_vector<int> expected_num_runs{3};
  // example-end reduce-by-key-env-determinism

  REQUIRE(unique_keys == expected_keys);
  REQUIRE(aggregates == expected_aggregates);
  REQUIRE(num_runs == expected_num_runs);
}

C2H_TEST("cub::DeviceReduce::ReduceByKey accepts stream", "[reduce][env]")
{
  // example-begin reduce-by-key-env-stream
  auto keys_in     = c2h::device_vector<int>{0, 0, 1, 1, 2, 2, 2};
  auto values_in   = c2h::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  auto unique_keys = c2h::device_vector<int>(3);
  auto aggregates  = c2h::device_vector<float>(3);
  auto num_runs    = c2h::device_vector<int>(1);

  // cudaStream_t legacy_stream = 0;
  // cuda::stream_ref stream_ref{legacy_stream};

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys.begin(),
    values_in.begin(),
    aggregates.begin(),
    num_runs.begin(),
    ::cuda::std::plus<float>{},
    keys_in.size(),
    env);

  c2h::device_vector<int> expected_keys{0, 1, 2};
  c2h::device_vector<float> expected_aggregates{3.0f, 7.0f, 18.0f};
  c2h::device_vector<int> expected_num_runs{3};
  // example-end reduce-by-key-env-stream

  REQUIRE(unique_keys == expected_keys);
  REQUIRE(aggregates == expected_aggregates);
  REQUIRE(num_runs == expected_num_runs);
}

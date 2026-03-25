// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceReduce::Reduce accepts run_to_run determinism requirements", "[reduce][env]")
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

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  stream.sync();
  // example-end reduce-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Sum accepts run_to_run determinism requirements", "[reduce][env]")
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

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{6.0f};
  stream.sync();
  // example-end sum-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Min accepts run_to_run determinism requirements", "[reduce][env]")
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

C2H_TEST("cub::DeviceReduce::Min accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin min-env-non-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{0.0f};
  // example-end min-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Min accepts stream", "[reduce][env]")
{
  // example-begin min-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{0.0f};
  stream.sync();
  // example-end min-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Max accepts run_to_run determinism requirements", "[reduce][env]")
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

C2H_TEST("cub::DeviceReduce::Max accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin max-env-non-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{3.0f};
  // example-end max-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::Max accepts stream", "[reduce][env]")
{
  // example-begin max-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{3.0f};
  stream.sync();
  // example-end max-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::ArgMin accepts run_to_run determinism requirements", "[reduce][env]")
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

C2H_TEST("cub::DeviceReduce::ArgMin accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin argmin-env-non-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-non-determinism

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

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error =
    cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected_min{0.0f};
  thrust::device_vector<cuda::std::int64_t> expected_index{3};
  stream.sync();
  // example-end argmin-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

C2H_TEST("cub::DeviceReduce::ArgMax accepts run_to_run determinism requirements", "[reduce][env]")
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

C2H_TEST("cub::DeviceReduce::ArgMax accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin argmax-env-non-determinism
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
  // example-end argmax-env-non-determinism

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

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

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

C2H_TEST("cub::DeviceReduce::TransformReduce accepts determinism requirements", "[reduce][env]")
{
  // example-begin transform-reduce-env-determinism
  auto op        = cuda::std::plus{};
  auto transform = cuda::std::negate{};
  auto input     = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output    = thrust::device_vector<float>(1);
  auto init      = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceReduce::TransformReduce(input.begin(), output.begin(), input.size(), op, transform, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::TransformReduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{-10.0f};
  // example-end transform-reduce-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::TransformReduce accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin transform-reduce-env-non-determinism
  auto op        = cuda::std::plus{};
  auto transform = cuda::std::negate{};
  auto input     = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output    = thrust::device_vector<float>(1);
  auto init      = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceReduce::TransformReduce(input.begin(), output.begin(), input.size(), op, transform, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::TransformReduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{-10.0f};
  // example-end transform-reduce-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::TransformReduce accepts stream", "[reduce][env]")
{
  // example-begin transform-reduce-env-stream
  auto op        = cuda::std::plus{};
  auto transform = cuda::std::negate{};
  auto input     = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output    = thrust::device_vector<float>(1);
  auto init      = 0.0f;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error =
    cub::DeviceReduce::TransformReduce(input.begin(), output.begin(), input.size(), op, transform, init, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::TransformReduce failed with status: " << error << std::endl;
  }

  thrust::device_vector<float> expected{-10.0f};
  stream.sync();
  // example-end transform-reduce-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

C2H_TEST("cub::DeviceReduce::ReduceByKey accepts run_to_run determinism requirements", "[reduce][env]")
{
  // example-begin reduce-by-key-env
  auto keys_in         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in       = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto unique_keys_out = thrust::device_vector<int>(5);
  auto aggregates_out  = thrust::device_vector<int>(5);
  auto num_runs_out    = thrust::device_vector<int>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys_out.begin(),
    values_in.begin(),
    aggregates_out.begin(),
    num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(keys_in.size()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  // example-end reduce-by-key-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_runs_out[0] == 5);
  unique_keys_out.resize(5);
  aggregates_out.resize(5);
  REQUIRE(unique_keys_out == expected_keys);
  REQUIRE(aggregates_out == expected_aggregates);
}

C2H_TEST("cub::DeviceReduce::ReduceByKey accepts not_guaranteed determinism requirements", "[reduce][env]")
{
  // example-begin reduce-by-key-env-non-determinism
  auto keys_in         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in       = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto unique_keys_out = thrust::device_vector<int>(5);
  auto aggregates_out  = thrust::device_vector<int>(5);
  auto num_runs_out    = thrust::device_vector<int>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys_out.begin(),
    values_in.begin(),
    aggregates_out.begin(),
    num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(keys_in.size()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  // example-end reduce-by-key-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_runs_out[0] == 5);
  unique_keys_out.resize(5);
  aggregates_out.resize(5);
  REQUIRE(unique_keys_out == expected_keys);
  REQUIRE(aggregates_out == expected_aggregates);
}

C2H_TEST("cub::DeviceReduce::ReduceByKey accepts stream", "[reduce][env]")
{
  // example-begin reduce-by-key-env-stream
  auto keys_in         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in       = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto unique_keys_out = thrust::device_vector<int>(5);
  auto aggregates_out  = thrust::device_vector<int>(5);
  auto num_runs_out    = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys_out.begin(),
    values_in.begin(),
    aggregates_out.begin(),
    num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(keys_in.size()),
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  stream.sync();
  // example-end reduce-by-key-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_runs_out[0] == 5);
  unique_keys_out.resize(5);
  aggregates_out.resize(5);
  REQUIRE(unique_keys_out == expected_keys);
  REQUIRE(aggregates_out == expected_aggregates);
}

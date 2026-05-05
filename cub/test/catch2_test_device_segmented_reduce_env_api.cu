// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSegmentedReduce::Sum accepts env with stream and determinism requirements",
         "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-sum-env
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto req_env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{req_env, stream_ref};

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-sum-env

  stream.sync();
  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum accepts stream", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-sum-env-stream
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceSegmentedReduce::Sum(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream_ref);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-sum-env-stream

  stream.sync();
  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum accepts run_to_run determinism requirements", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-sum-env-determinism
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-sum-env-determinism

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum accepts not_guaranteed determinism requirements", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-sum-env-non-determinism
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-sum-env-non-determinism

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Reduce env-based API", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-reduce-env
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::Reduce(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Reduce failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-reduce-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min env-based API", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-min-env
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{6, std::numeric_limits<int>::max(), 0};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Min failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-min-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Max env-based API", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-max-env
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{8, std::numeric_limits<int>::lowest(), 9};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Max failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-max-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin env-based API", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-argmin-env
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::ArgMin failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-argmin-env
  stream.sync();

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{1, 6}, {1, std::numeric_limits<int>::max()}, {2, 0}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax env-based API", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-argmax-env
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::ArgMax failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-argmax-env
  stream.sync();

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {1, std::numeric_limits<int>::lowest()}, {3, 9}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min accepts run_to_run determinism requirements", "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{6, std::numeric_limits<int>::max(), 0};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min accepts not_guaranteed determinism requirements", "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{6, std::numeric_limits<int>::max(), 0};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Max accepts run_to_run determinism requirements", "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{8, std::numeric_limits<int>::lowest(), 9};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Max accepts not_guaranteed determinism requirements", "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{8, std::numeric_limits<int>::lowest(), 9};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin accepts run_to_run determinism requirements", "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(error == cudaSuccess);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{1, 6}, {1, std::numeric_limits<int>::max()}, {2, 0}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin accepts not_guaranteed determinism requirements",
         "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(error == cudaSuccess);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{1, 6}, {1, std::numeric_limits<int>::max()}, {2, 0}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax accepts run_to_run determinism requirements", "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(error == cudaSuccess);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {1, std::numeric_limits<int>::lowest()}, {3, 9}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax accepts not_guaranteed determinism requirements",
         "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);

  REQUIRE(error == cudaSuccess);

  thrust::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {1, std::numeric_limits<int>::lowest()}, {3, 9}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Reduce accepts run_to_run determinism requirements", "[segmented_reduce][env]")
{
  // example-begin segmented-reduce-reduce-env-determinism
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceSegmentedReduce::Reduce(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Reduce failed with status: " << error << '\n';
  }
  // example-end segmented-reduce-reduce-env-determinism

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Reduce accepts not_guaranteed determinism requirements",
         "[segmented_reduce][env]")
{
  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceSegmentedReduce::Reduce(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, ::cuda::std::plus<>{}, 0, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Reduce failed with status: " << error << '\n';
  }

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}
C2H_TEST("cub::DeviceSegmentedReduce::Reduce fixed-size env-based API", "[segmented_reduce][env]")
{
  // example-begin fixed-size-segmented-reduce-reduce-env
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::Reduce(
    d_in.begin(), d_out.begin(), num_segments, segment_size, ::cuda::std::plus<>{}, 0, env);
  thrust::device_vector<int> expected{21, 8};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Reduce (fixed-size) failed with status: " << error << '\n';
  }
  // example-end fixed-size-segmented-reduce-reduce-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum fixed-size env-based API", "[segmented_reduce][env]")
{
  // example-begin fixed-size-segmented-reduce-sum-env
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, segment_size, env);
  thrust::device_vector<int> expected{21, 8};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum (fixed-size) failed with status: " << error << '\n';
  }
  // example-end fixed-size-segmented-reduce-sum-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min fixed-size env-based API", "[segmented_reduce][env]")
{
  // example-begin fixed-size-segmented-reduce-min-env
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::Min(d_in.begin(), d_out.begin(), num_segments, segment_size, env);
  thrust::device_vector<int> expected{6, 0};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Min (fixed-size) failed with status: " << error << '\n';
  }
  // example-end fixed-size-segmented-reduce-min-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::Max fixed-size env-based API", "[segmented_reduce][env]")
{
  // example-begin fixed-size-segmented-reduce-max-env
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<int> d_out(2);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::Max(d_in.begin(), d_out.begin(), num_segments, segment_size, env);
  thrust::device_vector<int> expected{8, 5};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Max (fixed-size) failed with status: " << error << '\n';
  }
  // example-end fixed-size-segmented-reduce-max-env
  stream.sync();

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin fixed-size env-based API", "[segmented_reduce][env]")
{
  // example-begin fixed-size-segmented-reduce-argmin-env
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::ArgMin(d_in.begin(), d_out.begin(), num_segments, segment_size, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::ArgMin (fixed-size) failed with status: " << error << '\n';
  }
  // example-end fixed-size-segmented-reduce-argmin-env
  stream.sync();

  thrust::device_vector<cuda::std::pair<int, int>> expected{{1, 6}, {2, 0}};
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax fixed-size env-based API", "[segmented_reduce][env]")
{
  // example-begin fixed-size-segmented-reduce-argmax-env
  int num_segments = 2;
  int segment_size = 3;
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0};
  thrust::device_vector<cuda::std::pair<int, int>> d_out(2);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = ::cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedReduce::ArgMax(d_in.begin(), d_out.begin(), num_segments, segment_size, env);

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::ArgMax (fixed-size) failed with status: " << error << '\n';
  }
  // example-end fixed-size-segmented-reduce-argmax-env
  stream.sync();

  thrust::device_vector<cuda::std::pair<int, int>> expected{{0, 8}, {0, 5}};
  REQUIRE(d_out == expected);
}

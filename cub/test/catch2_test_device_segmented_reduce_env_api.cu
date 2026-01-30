// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

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

  auto req_env               = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto env = ::cuda::std::execution::env{req_env, stream_ref};

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << std::endl;
  }
  // example-end segmented-reduce-sum-env

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

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};

  auto error = cub::DeviceSegmentedReduce::Sum(
    d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, stream_ref);
  thrust::device_vector<int> expected{21, 0, 17};

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << std::endl;
  }
  // example-end segmented-reduce-sum-env-stream

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
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << std::endl;
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
    std::cerr << "cub::DeviceSegmentedReduce::Sum failed with status: " << error << std::endl;
  }
  // example-end segmented-reduce-sum-env-non-determinism

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}

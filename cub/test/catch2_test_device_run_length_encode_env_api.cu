// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceRunLengthEncode::Encode accepts env with stream", "[run_length_encode][env]")
{
  // example-begin encode-env
  auto input        = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto unique_out   = thrust::device_vector<int>(8);
  auto counts_out   = thrust::device_vector<int>(8);
  auto num_runs_out = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceRunLengthEncode::Encode(
    input.begin(), unique_out.begin(), counts_out.begin(), num_runs_out.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthEncode::Encode failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_unique{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_counts{1, 2, 1, 3, 1};
  thrust::device_vector<int> expected_num_runs{5};
  // example-end encode-env

  REQUIRE(error == cudaSuccess);
  unique_out.resize(num_runs_out[0]);
  counts_out.resize(num_runs_out[0]);
  REQUIRE(unique_out == expected_unique);
  REQUIRE(counts_out == expected_counts);
  REQUIRE(num_runs_out == expected_num_runs);
}

C2H_TEST("cub::DeviceRunLengthEncode::NonTrivialRuns accepts env with stream", "[run_length_encode][env]")
{
  // example-begin non-trivial-runs-env
  auto input        = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto offsets_out  = thrust::device_vector<int>(8);
  auto lengths_out  = thrust::device_vector<int>(8);
  auto num_runs_out = thrust::device_vector<int>(1);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceRunLengthEncode::NonTrivialRuns(
    input.begin(), offsets_out.begin(), lengths_out.begin(), num_runs_out.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceRunLengthEncode::NonTrivialRuns failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_offsets{1, 4};
  thrust::device_vector<int> expected_lengths{2, 3};
  thrust::device_vector<int> expected_num_runs{2};
  // example-end non-trivial-runs-env

  REQUIRE(error == cudaSuccess);
  offsets_out.resize(num_runs_out[0]);
  lengths_out.resize(num_runs_out[0]);
  REQUIRE(offsets_out == expected_offsets);
  REQUIRE(lengths_out == expected_lengths);
  REQUIRE(num_runs_out == expected_num_runs);
}

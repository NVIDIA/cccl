// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_partition.cuh>

#include <thrust/device_vector.h>

#include <iostream>

#include "catch2_test_device_select_common.cuh"
#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DevicePartition::If accepts env with stream", "[partition][env]")
{
  // example-begin partition-if-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto output       = thrust::device_vector<int>(8);
  auto num_selected = thrust::device_vector<int>(1);
  less_than_t<int> le{5};

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DevicePartition::If(input.begin(), output.begin(), num_selected.begin(), input.size(), le, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DevicePartition::If failed with status: " << error << std::endl;
  }

  // Selected items (< 5) at front, unselected items (>= 5) at back in reverse order
  thrust::device_vector<int> expected_output{1, 2, 3, 4, 8, 7, 6, 5};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end partition-if-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DevicePartition::Flagged accepts env with stream", "[partition][env]")
{
  // example-begin partition-flagged-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto flags        = thrust::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto output       = thrust::device_vector<int>(8);
  auto num_selected = thrust::device_vector<int>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DevicePartition::Flagged(
    input.begin(), flags.begin(), output.begin(), num_selected.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DevicePartition::Flagged failed with status: " << error << std::endl;
  }

  // Selected items (flagged) at front, unselected items at back in reverse order
  thrust::device_vector<int> expected_output{1, 4, 6, 7, 8, 5, 3, 2};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end partition-flagged-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

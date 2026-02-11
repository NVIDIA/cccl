// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>

#include <iostream>

#include "catch2_test_device_select_common.cuh"
#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSelect::If accepts env with stream", "[select][env]")
{
  // example-begin select-if-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto output       = thrust::device_vector<int>(4);
  auto num_selected = thrust::device_vector<int>(1);
  less_than_t<int> le{5};

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSelect::If(input.begin(), output.begin(), num_selected.begin(), input.size(), le, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::If failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_output{1, 2, 3, 4};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-if-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

C2H_TEST("cub::DeviceSelect::Flagged accepts env with stream", "[select][env]")
{
  // example-begin select-flagged-env
  auto input        = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto flags        = thrust::device_vector<char>{1, 0, 0, 1, 0, 1, 1, 0};
  auto output       = thrust::device_vector<int>(4);
  auto num_selected = thrust::device_vector<int>(1);

  cudaStream_t legacy_stream = 0;
  cuda::stream_ref stream_ref{legacy_stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error =
    cub::DeviceSelect::Flagged(input.begin(), flags.begin(), output.begin(), num_selected.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSelect::Flagged failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_output{1, 4, 6, 7};
  thrust::device_vector<int> expected_num_selected{4};
  // example-end select-flagged-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected_output);
  REQUIRE(num_selected == expected_num_selected);
}

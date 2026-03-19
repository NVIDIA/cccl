// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceMerge::MergeKeys accepts env with stream", "[merge][env]")
{
  // example-begin merge-keys-env
  auto keys1  = thrust::device_vector<int>{0, 2, 5};
  auto keys2  = thrust::device_vector<int>{0, 3, 3, 4};
  auto result = thrust::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceMerge::MergeKeys(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    cuda::std::less<>{},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMerge::MergeKeys failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  // example-end merge-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result == expected);
}

C2H_TEST("cub::DeviceMerge::MergePairs accepts env with stream", "[merge][env]")
{
  // example-begin merge-pairs-env
  auto keys1   = thrust::device_vector<int>{0, 2, 5};
  auto values1 = thrust::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = thrust::device_vector<int>{0, 3, 3, 4};
  auto values2 = thrust::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = thrust::device_vector<int>(7);
  auto result_values = thrust::device_vector<char>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceMerge::MergePairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    cuda::std::less<>{},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMerge::MergePairs failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  thrust::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  // example-end merge-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

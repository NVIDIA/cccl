// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/functional>
#include <cuda/stream>

#include <iostream>

#include "cub_test_macros.h"

CUB_TEST("cub::detail::DeviceSetOps::SetDifference accepts an environment", "[set_ops][env]", CUB_SMALL)
{
  // example-begin set-difference-env
  auto keys1        = thrust::device_vector<int>{0, 2, 4, 5, 7};
  auto keys2        = thrust::device_vector<int>{1, 2, 3, 5};
  auto result       = thrust::device_vector<int>(keys1.size() + keys2.size(), thrust::no_init);
  auto num_selected = thrust::device_vector<int>(1, thrust::no_init);

  const cuda::stream stream{cuda::devices[0]};

  auto error = cub::detail::DeviceSetOps::SetDifference(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num_selected.begin(),
    cuda::std::less<>{},
    cuda::stream_ref{stream});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::detail::DeviceSetOps::SetDifference failed with status: " << error << '\n';
  }

  // keys present in keys1 but not in keys2
  const thrust::device_vector<int> expected{0, 4, 7};
  // example-end set-difference-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  result.resize(num_selected[0]);
  REQUIRE(result == expected);
}

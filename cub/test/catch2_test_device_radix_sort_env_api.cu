// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceRadixSort::SortPairs accepts determinism requirements", "[radix_sort][env]")
{
  // example-begin radix-sort-pairs-env-determinism
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);

  auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);

  auto error = cub::DeviceRadixSort::SortPairs(
    keys_in.data().get(),
    keys_out.data().get(),
    values_in.data().get(),
    values_out.data().get(),
    static_cast<int>(keys_in.size()),
    0,
    sizeof(int) * 8,
    env);

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  thrust::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  // example-end radix-sort-pairs-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

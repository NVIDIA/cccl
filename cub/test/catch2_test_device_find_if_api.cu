// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <c2h/catch2_test_helper.h>

// example-begin find-if-predicate
struct is_greater_than_t
{
  int threshold;
  __host__ __device__ bool operator()(int value) const
  {
    return value > threshold;
  }
};
// example-end find-if-predicate

C2H_TEST("cub::DeviceFind::FindIf works with int data elements", "[find][device]")
{
  // example-begin device-find-if
  constexpr int num_items         = 8;
  thrust::device_vector<int> d_in = {0, 1, 2, 3, 4, 5, 6, 7};
  thrust::device_vector<int> d_out(1, thrust::no_init);
  is_greater_than_t predicate{4};

  size_t temp_storage_bytes = 0;
  cub::DeviceFind::FindIf(nullptr, temp_storage_bytes, d_in.begin(), d_out.begin(), predicate, num_items);

  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);

  cub::DeviceFind::FindIf(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    predicate,
    num_items);

  int expected = 5;
  // example-end device-find-if

  REQUIRE(d_out[0] == expected);
}

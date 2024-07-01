// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_merge.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "catch2_test_helper.h"

CUB_TEST("DeviceMerge::MergeKeys API example", "[merge][device]")
{
  // example-begin merge-keys
  thrust::device_vector<int> keys1{0, 2, 5};
  thrust::device_vector<int> keys2{0, 3, 3, 4};
  thrust::device_vector<int> result(7);

  // 1) Get temp storage size
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMerge::MergeKeys(
    nullptr,
    temp_storage_bytes,
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin());

  // 2) Allocate temp storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes);

  // 3) Perform merge operation
  cub::DeviceMerge::MergeKeys(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin());

  CHECK(result == thrust::host_vector<int>{0, 0, 2, 3, 3, 4, 5});
  // example-end merge-keys
}

CUB_TEST("DeviceMerge::MergePairs API example", "[merge][device]")
{
  // example-begin merge-pairs
  thrust::device_vector<int> keys1{0, 2, 5};
  thrust::device_vector<char> values1{'a', 'b', 'c'};
  thrust::device_vector<int> keys2{0, 3, 3, 4};
  thrust::device_vector<char> values2{'A', 'B', 'C', 'D'};
  thrust::device_vector<int> result_keys(7);
  thrust::device_vector<char> result_values(7);

  // 1) Get temp storage size
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMerge::MergePairs(
    nullptr,
    temp_storage_bytes,
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin());

  // 2) Allocate temp storage
  thrust::device_vector<char> temp_storage(temp_storage_bytes);

  // 3) Perform merge operation
  cub::DeviceMerge::MergePairs(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage_bytes,
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin());

  CHECK(result_keys == thrust::host_vector<int>{0, 0, 2, 3, 3, 4, 5});
  CHECK(result_values == thrust::host_vector<char>{'a', 'A', 'b', 'B', 'C', 'D', 'c'});
  // example-end merge-pairs
}

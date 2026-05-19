// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_merge_sort.cuh>

#include <thrust/device_vector.h>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceMergeSort::SortPairs non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys(1);
  thrust::device_vector<int> values(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortPairs(nullptr, temp_storage_bytes, keys.begin(), values.begin(), 1, cuda::std::less<>{});
}

C2H_TEST("cub::DeviceMergeSort::SortPairsCopy non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys_in(1);
  thrust::device_vector<int> values_in(1);
  thrust::device_vector<int> keys_out(1);
  thrust::device_vector<int> values_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortPairsCopy(
    nullptr,
    temp_storage_bytes,
    keys_in.begin(),
    values_in.begin(),
    keys_out.begin(),
    values_out.begin(),
    1,
    cuda::std::less<>{});
}

C2H_TEST("cub::DeviceMergeSort::SortKeys non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, keys.begin(), 1, cuda::std::less<>{});
}

C2H_TEST("cub::DeviceMergeSort::SortKeysCopy non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys_in(1);
  thrust::device_vector<int> keys_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(
    nullptr, temp_storage_bytes, keys_in.begin(), keys_out.begin(), 1, cuda::std::less<>{});
}

C2H_TEST("cub::DeviceMergeSort::StableSortPairs non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys(1);
  thrust::device_vector<int> values(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::StableSortPairs(
    nullptr, temp_storage_bytes, keys.begin(), values.begin(), 1, cuda::std::less<>{});
}

C2H_TEST("cub::DeviceMergeSort::StableSortKeys non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::StableSortKeys(nullptr, temp_storage_bytes, keys.begin(), 1, cuda::std::less<>{});
}

C2H_TEST("cub::DeviceMergeSort::StableSortKeysCopy non-env API", "[merge_sort][device]")
{
  thrust::device_vector<int> keys_in(1);
  thrust::device_vector<int> keys_out(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::StableSortKeysCopy(
    nullptr, temp_storage_bytes, keys_in.begin(), keys_out.begin(), 1, cuda::std::less<>{});
}

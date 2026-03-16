// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/device_vector.h>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceMergeSort::SortPairs env-based API", "[merge_sort][env]")
{
  // example-begin sort-pairs-env
  auto d_keys   = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  auto error = cub::DeviceMergeSort::SortPairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::SortPairs failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  thrust::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  // example-end sort-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

C2H_TEST("cub::DeviceMergeSort::SortKeys env-based API", "[merge_sort][env]")
{
  // example-begin sort-keys-env
  auto d_keys = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  auto error =
    cub::DeviceMergeSort::SortKeys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::SortKeys failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys == expected_keys);
}

C2H_TEST("cub::DeviceMergeSort::StableSortPairs env-based API", "[merge_sort][env]")
{
  // example-begin stable-sort-pairs-env
  auto d_keys   = thrust::device_vector<int>{8, 6, 6, 5, 3, 0, 9};
  auto d_values = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  auto error = cub::DeviceMergeSort::StableSortPairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::StableSortPairs failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  thrust::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  // example-end stable-sort-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

C2H_TEST("cub::DeviceMergeSort::StableSortKeys env-based API", "[merge_sort][env]")
{
  // example-begin stable-sort-keys-env
  auto d_keys = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  auto error =
    cub::DeviceMergeSort::StableSortKeys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{});
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::StableSortKeys failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end stable-sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys == expected_keys);
}

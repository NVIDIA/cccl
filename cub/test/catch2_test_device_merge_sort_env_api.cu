// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

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
    std::cerr << "cub::DeviceMergeSort::SortPairs failed with status: " << error << '\n';
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
    std::cerr << "cub::DeviceMergeSort::SortKeys failed with status: " << error << '\n';
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
    std::cerr << "cub::DeviceMergeSort::StableSortPairs failed with status: " << error << '\n';
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
    std::cerr << "cub::DeviceMergeSort::StableSortKeys failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end stable-sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys == expected_keys);
}

C2H_TEST("cub::DeviceMergeSort::SortPairsCopy env-based API", "[merge_sort][env]")
{
  // example-begin sort-pairs-copy-env
  auto d_keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out   = thrust::device_vector<int>(7);
  auto d_values_out = thrust::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceMergeSort::SortPairsCopy(
    d_keys_in.data().get(),
    d_values_in.data().get(),
    d_keys_out.data().get(),
    d_values_out.data().get(),
    static_cast<int>(d_keys_in.size()),
    cuda::std::less<int>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::SortPairsCopy failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  thrust::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  // example-end sort-pairs-copy-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

C2H_TEST("cub::DeviceMergeSort::SortKeysCopy env-based API", "[merge_sort][env]")
{
  // example-begin sort-keys-copy-env
  auto d_keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = thrust::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceMergeSort::SortKeysCopy(
    d_keys_in.data().get(),
    d_keys_out.data().get(),
    static_cast<int>(d_keys_in.size()),
    cuda::std::less<int>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::SortKeysCopy failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end sort-keys-copy-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys_out == expected_keys);
}

C2H_TEST("cub::DeviceMergeSort::StableSortKeysCopy env-based API", "[merge_sort][env]")
{
  // example-begin stable-sort-keys-copy-env
  auto d_keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = thrust::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceMergeSort::StableSortKeysCopy(
    d_keys_in.data().get(),
    d_keys_out.data().get(),
    static_cast<int>(d_keys_in.size()),
    cuda::std::less<int>{},
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMergeSort::StableSortKeysCopy failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  // example-end stable-sort-keys-copy-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_keys_out == expected_keys);
}

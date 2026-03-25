// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSegmentedRadixSort::SortPairs env with stream", "[segmented_radix_sort][env]")
{
  // example-begin segmented-radix-sort-pairs-env
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);
  auto offsets    = thrust::device_vector<int>{0, 3, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedRadixSort::SortPairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    cuda::std::int64_t{3},
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedRadixSort::SortPairs failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  thrust::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  // example-end segmented-radix-sort-pairs-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceSegmentedRadixSort::SortPairsDescending env with stream", "[segmented_radix_sort][env]")
{
  // example-begin segmented-radix-sort-pairs-descending-env
  auto keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = thrust::device_vector<int>(7);
  auto values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = thrust::device_vector<int>(7);
  auto offsets    = thrust::device_vector<int>{0, 3, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedRadixSort::SortPairsDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    cuda::std::int64_t{3},
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedRadixSort::SortPairsDescending failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  thrust::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  // example-end segmented-radix-sort-pairs-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceSegmentedRadixSort::SortKeys env with stream", "[segmented_radix_sort][env]")
{
  // example-begin segmented-radix-sort-keys-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);
  auto offsets  = thrust::device_vector<int>{0, 3, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedRadixSort::SortKeys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    cuda::std::int64_t{3},
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedRadixSort::SortKeys failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  // example-end segmented-radix-sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
}

C2H_TEST("cub::DeviceSegmentedRadixSort::SortKeysDescending env with stream", "[segmented_radix_sort][env]")
{
  // example-begin segmented-radix-sort-keys-descending-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);
  auto offsets  = thrust::device_vector<int>{0, 3, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedRadixSort::SortKeysDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    cuda::std::int64_t{3},
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedRadixSort::SortKeysDescending failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  // example-end segmented-radix-sort-keys-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
}

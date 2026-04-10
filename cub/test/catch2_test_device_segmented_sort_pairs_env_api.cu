// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSegmentedSort::StableSortPairs env-based API", "[segmented_sort][pairs][env]")
{
  // example-begin stable-sort-pairs-env
  auto keys_in       = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out      = thrust::device_vector<int>(7);
  auto values_in     = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out    = thrust::device_vector<int>(7);
  auto offsets_begin = thrust::device_vector<int>{0, 3};
  auto offsets_end   = thrust::device_vector<int>{3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortPairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets_begin.data()),
    thrust::raw_pointer_cast(offsets_end.data()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::StableSortPairs failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  thrust::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  // example-end stable-sort-pairs-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortPairsDescending env-based API", "[segmented_sort][pairs][env]")
{
  // example-begin stable-sort-pairs-descending-env
  auto keys_in       = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out      = thrust::device_vector<int>(7);
  auto values_in     = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out    = thrust::device_vector<int>(7);
  auto offsets_begin = thrust::device_vector<int>{0, 3};
  auto offsets_end   = thrust::device_vector<int>{3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortPairsDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets_begin.data()),
    thrust::raw_pointer_cast(offsets_end.data()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::StableSortPairsDescending failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  thrust::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  // example-end stable-sort-pairs-descending-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortPairs DoubleBuffer env-based API", "[segmented_sort][pairs][env]")
{
  // example-begin stable-sort-pairs-db-env
  auto keys_buf0     = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1     = thrust::device_vector<int>(7);
  auto values_buf0   = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1   = thrust::device_vector<int>(7);
  auto offsets_begin = thrust::device_vector<int>{0, 3};
  auto offsets_end   = thrust::device_vector<int>{3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortPairs(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets_begin.data()),
    thrust::raw_pointer_cast(offsets_end.data()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::StableSortPairs (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  thrust::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  // example-end stable-sort-pairs-db-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  thrust::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortPairsDescending DoubleBuffer env-based API",
         "[segmented_sort][pairs][env]")
{
  // example-begin stable-sort-pairs-descending-db-env
  auto keys_buf0     = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1     = thrust::device_vector<int>(7);
  auto values_buf0   = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1   = thrust::device_vector<int>(7);
  auto offsets_begin = thrust::device_vector<int>{0, 3};
  auto offsets_end   = thrust::device_vector<int>{3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortPairsDescending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets_begin.data()),
    thrust::raw_pointer_cast(offsets_end.data()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr
      << "cub::DeviceSegmentedSort::StableSortPairsDescending (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  thrust::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  // example-end stable-sort-pairs-descending-db-env
  stream.sync();

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  thrust::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

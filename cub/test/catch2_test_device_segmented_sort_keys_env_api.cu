// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceSegmentedSort::SortKeys env-based API", "[segmented_sort][keys][env]")
{
  // example-begin sort-keys-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);
  auto offsets  = thrust::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortKeys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::SortKeys failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  // example-end sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::SortKeysDescending env-based API", "[segmented_sort][keys][env]")
{
  // example-begin sort-keys-descending-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);
  auto offsets  = thrust::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortKeysDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::SortKeysDescending failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  // example-end sort-keys-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::SortKeys DoubleBuffer env-based API", "[segmented_sort][keys][env]")
{
  // example-begin sort-keys-db-env
  auto keys_buf0 = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = thrust::device_vector<int>(7);
  auto offsets   = thrust::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortKeys(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::SortKeys (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  // example-end sort-keys-db-env

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::SortKeysDescending DoubleBuffer env-based API", "[segmented_sort][keys][env]")
{
  // example-begin sort-keys-descending-db-env
  auto keys_buf0 = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = thrust::device_vector<int>(7);
  auto offsets   = thrust::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortKeysDescending(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::SortKeysDescending (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  // example-end sort-keys-descending-db-env

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortKeys env-based API", "[segmented_sort][keys][env]")
{
  // example-begin stable-sort-keys-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);
  auto offsets  = thrust::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortKeys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::StableSortKeys failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  // example-end stable-sort-keys-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortKeysDescending env-based API", "[segmented_sort][keys][env]")
{
  // example-begin stable-sort-keys-descending-env
  auto keys_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = thrust::device_vector<int>(7);
  auto offsets  = thrust::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortKeysDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::StableSortKeysDescending failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  // example-end stable-sort-keys-descending-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(keys_out == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortKeys DoubleBuffer env-based API", "[segmented_sort][keys][env]")
{
  // example-begin stable-sort-keys-db-env
  auto keys_buf0 = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = thrust::device_vector<int>(7);
  auto offsets   = thrust::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortKeys(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceSegmentedSort::StableSortKeys (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  // example-end stable-sort-keys-db-env

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

C2H_TEST("cub::DeviceSegmentedSort::StableSortKeysDescending DoubleBuffer env-based API", "[segmented_sort][keys][env]")
{
  // example-begin stable-sort-keys-descending-db-env
  auto keys_buf0 = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = thrust::device_vector<int>(7);
  auto offsets   = thrust::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortKeysDescending(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);
  if (error != cudaSuccess)
  {
    std::cerr
      << "cub::DeviceSegmentedSort::StableSortKeysDescending (DoubleBuffer) failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  // example-end stable-sort-keys-descending-db-env

  REQUIRE(error == cudaSuccess);
  thrust::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

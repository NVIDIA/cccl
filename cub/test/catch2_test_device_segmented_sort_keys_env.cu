// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("DeviceSegmentedSort::SortKeys uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortKeys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::SortKeysDescending uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortKeysDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeys uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortKeys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeysDescending uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::StableSortKeysDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::SortKeys DoubleBuffer uses environment", "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

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

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

C2H_TEST("DeviceSegmentedSort::SortKeysDescending DoubleBuffer uses environment", "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

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

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeys DoubleBuffer uses environment", "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

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

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeysDescending DoubleBuffer uses environment",
         "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

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

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Note: This test does not use the DECLARE_LAUNCH_WRAPPER / stream_registry_factory_t pattern.
// SegmentedSort's dispatch performs a D->H memcpy mid-dispatch via launcher_factory.MemcpyAsync()
// (dispatch_segmented_sort.cuh:885,1125) to read partition group sizes back to the host.
// stream_registry_factory_t does not implement MemcpyAsync, so the parameterized launch test
// (lid 0:1:2) cannot be used here.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("DeviceSegmentedSort::SortPairs uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortPairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairsDescending uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortPairsDescending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairs uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

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
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairsDescending uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

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
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairs DoubleBuffer uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortPairs(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairsDescending DoubleBuffer uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceSegmentedSort::SortPairsDescending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairs DoubleBuffer uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

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
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairsDescending DoubleBuffer uses environment",
         "[segmented_sort][pairs][device]")
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

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
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  REQUIRE(error == cudaSuccess);
  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

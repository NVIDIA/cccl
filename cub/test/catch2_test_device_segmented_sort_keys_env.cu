// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortKeysDescending, sort_keys_descending);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortKeys, stable_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortKeysDescending, stable_sort_keys_descending);

// %PARAM% TEST_LAUNCH lid 0:1

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceSegmentedSort::SortKeys works with default environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeys(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

TEST_CASE("DeviceSegmentedSort::SortKeysDescending works with default environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeysDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

TEST_CASE("DeviceSegmentedSort::StableSortKeys works with default environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeys(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

TEST_CASE("DeviceSegmentedSort::StableSortKeysDescending works with default environment",
          "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeysDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<cuda::std::int64_t>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

TEST_CASE("DeviceSegmentedSort::SortKeys DoubleBuffer works with default environment", "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::SortKeys(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

TEST_CASE("DeviceSegmentedSort::SortKeysDescending DoubleBuffer works with default environment",
          "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::SortKeysDescending(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

TEST_CASE("DeviceSegmentedSort::StableSortKeys DoubleBuffer works with default environment",
          "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::StableSortKeys(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

TEST_CASE("DeviceSegmentedSort::StableSortKeysDescending DoubleBuffer works with default environment",
          "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedSort::StableSortKeysDescending(
            d_keys,
            static_cast<cuda::std::int64_t>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

#endif

C2H_TEST("DeviceSegmentedSort::SortKeys uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys(thrust::raw_pointer_cast(keys_in.data()),
            thrust::raw_pointer_cast(keys_out.data()),
            static_cast<cuda::std::int64_t>(keys_in.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1,
            env);

  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::SortKeysDescending uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeys uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeys(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeysDescending uses environment", "[segmented_sort][keys][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<cuda::std::int64_t>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected);
}

C2H_TEST("DeviceSegmentedSort::SortKeys DoubleBuffer uses environment", "[segmented_sort][keys][device]")
{
  auto keys_buf0 = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1 = c2h::device_vector<int>(7);
  auto offsets   = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys(d_keys,
            static_cast<int>(keys_buf0.size()),
            2,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(offsets.data()) + 1,
            env);

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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeys(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_keys_descending(
    d_keys,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> result(d_keys.Current(), d_keys.Current() + 7);
  REQUIRE(result == expected);
}

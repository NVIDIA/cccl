// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>

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

template <int BlockThreads>
struct segmented_sort_tuning
{
  _CCCL_API constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::segmented_sort::segmented_sort_policy
  {
    return {
      cub::detail::segmented_sort::segmented_radix_sort_policy{
        BlockThreads, 1, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::RADIX_RANK_BASIC, cub::BLOCK_SCAN_WARP_SCANS, 4},
      cub::detail::segmented_sort::sub_warp_merge_sort_policy{
        BlockThreads, 4, 1, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT},
      cub::detail::segmented_sort::sub_warp_merge_sort_policy{
        BlockThreads, 32, 1, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT},
      1000000};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

C2H_TEST("DeviceSegmentedSort::SortKeys can be tuned", "[segmented_sort][keys][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  sort_keys(thrust::raw_pointer_cast(keys_in.data()),
            thrust::raw_pointer_cast(keys_out.data()),
            static_cast<int>(keys_in.size()),
            1,
            thrust::raw_pointer_cast(begin_offsets.data()),
            end_offsets,
            env);

  c2h::device_vector<int> expected{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedSort::SortKeysDescending can be tuned", "[segmented_sort][keys][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected{9, 8, 7, 6, 5, 3, 0};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeys can be tuned", "[segmented_sort][keys][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  stable_sort_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedSort::StableSortKeysDescending can be tuned", "[segmented_sort][keys][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  stable_sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected{9, 8, 7, 6, 5, 3, 0};
  REQUIRE(keys_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

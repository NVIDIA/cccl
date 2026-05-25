// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortPairs, stable_sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortPairsDescending, stable_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortPairsDescending, sort_pairs_descending);

// %PARAM% TEST_LAUNCH lid 0:1

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceSegmentedSort::StableSortPairs works with default environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortPairs(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<int>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedSort::StableSortPairsDescending works with default environment",
          "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortPairsDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<int>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairs nonstable works with default environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairs(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<int>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable works with default environment",
          "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairsDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<int>(keys_in.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairs nonstable DoubleBuffer works with default environment",
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

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairs(
      d_keys,
      d_values,
      static_cast<int>(keys_buf0.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable DoubleBuffer works with default environment",
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

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairsDescending(
      d_keys,
      d_values,
      static_cast<int>(keys_buf0.size()),
      2,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

#endif

TEST_CASE("DeviceSegmentedSort::SortPairs nonstable uses custom stream", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  stream.sync();

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable uses custom stream", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  cuda::stream stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  stream.sync();

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairs nonstable DoubleBuffer uses custom stream", "[segmented_sort][pairs][device]")
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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(d_keys,
             d_values,
             static_cast<int>(keys_buf0.size()),
             2,
             thrust::raw_pointer_cast(offsets.data()),
             thrust::raw_pointer_cast(offsets.data()) + 1,
             env);

  stream.sync();

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable DoubleBuffer uses custom stream",
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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  stream.sync();

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairs uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortPairs(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_pairs(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::StableSortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  stable_sort_pairs_descending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairs nonstable uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairsDescending nonstable uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairs nonstable DoubleBuffer uses environment", "[segmented_sort][pairs][device]")
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(d_keys,
             d_values,
             static_cast<int>(keys_buf0.size()),
             2,
             thrust::raw_pointer_cast(offsets.data()),
             thrust::raw_pointer_cast(offsets.data()) + 1,
             env);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedSort::SortPairsDescending nonstable DoubleBuffer uses environment",
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

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf0.size()),
      static_cast<::cuda::std::int64_t>(2),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(offsets.data()) + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf0.size()),
    2,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    env);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
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

C2H_TEST("DeviceSegmentedSort::SortPairs can be tuned", "[segmented_sort][pairs][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedSort::SortPairsDescending can be tuned", "[segmented_sort][pairs][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairs can be tuned", "[segmented_sort][pairs][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  stable_sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceSegmentedSort::StableSortPairsDescending can be tuned", "[segmented_sort][pairs][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

  auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  stable_sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    1,
    thrust::raw_pointer_cast(begin_offsets.data()),
    end_offsets,
    env);

  c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

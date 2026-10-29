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
#include <cuda/__functional/call_or.h>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::StableSortPairs, stable_sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::StableSortPairsDescending, stable_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedSort::SortPairsDescending, sort_pairs_descending);

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortPairs, stable_sort_pairs_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortPairsDescending, stable_sort_pairs_descending_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortPairs, sort_pairs_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::SortPairsDescending, sort_pairs_descending_two_phase);

// %PARAM% TEST_LAUNCH lid 0:1

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceSegmentedSort::StableSortPairs works with default environment",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::StableSortPairsDescending works with default environment",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairs nonstable works with default environment",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable works with default environment",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairs nonstable DoubleBuffer works with default environment",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable DoubleBuffer works with default environment",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

#endif

CUB_TEST_CASE("DeviceSegmentedSort::SortPairs nonstable uses custom stream",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  const cuda::stream stream = c2h::make_current_device_stream();

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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable uses custom stream",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 7};

  const cuda::stream stream = c2h::make_current_device_stream();

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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairs nonstable DoubleBuffer uses custom stream",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  const cuda::stream stream = c2h::make_current_device_stream();

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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST_CASE("DeviceSegmentedSort::SortPairsDescending nonstable DoubleBuffer uses custom stream",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_buf1   = c2h::device_vector<int>(7);
  auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_buf1 = c2h::device_vector<int>(7);
  auto offsets     = c2h::device_vector<int>{0, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  const cuda::stream stream = c2h::make_current_device_stream();

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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("DeviceSegmentedSort::StableSortPairs uses environment", "[segmented_sort][pairs][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("DeviceSegmentedSort::StableSortPairsDescending uses environment", "[segmented_sort][pairs][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("DeviceSegmentedSort::StableSortPairs DoubleBuffer uses environment",
         "[segmented_sort][pairs][device]",
         CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("DeviceSegmentedSort::StableSortPairsDescending DoubleBuffer uses environment",
         "[segmented_sort][pairs][device]",
         CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("DeviceSegmentedSort::SortPairs nonstable uses environment", "[segmented_sort][pairs][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("DeviceSegmentedSort::SortPairsDescending nonstable uses environment",
         "[segmented_sort][pairs][device]",
         CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

CUB_TEST("DeviceSegmentedSort::SortPairs nonstable DoubleBuffer uses environment",
         "[segmented_sort][pairs][device]",
         CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  const c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST("DeviceSegmentedSort::SortPairsDescending nonstable DoubleBuffer uses environment",
         "[segmented_sort][pairs][device]",
         CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  const c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  const c2h::device_vector<int> result_keys(d_keys.Current(), d_keys.Current() + 7);
  const c2h::device_vector<int> result_values(d_values.Current(), d_values.Current() + 7);
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

template <int BlockThreads>
struct segmented_sort_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::SegmentedSortPolicy
  {
    return cub::SegmentedSortPolicy{
      cub::SegmentedSortRadixSortPolicy{
        BlockThreads, 1, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::RADIX_RANK_BASIC, cub::BLOCK_SCAN_WARP_SCANS, 4},
      cub::SegmentedSortSubWarpMergeSortPolicy{
        BlockThreads, 32, 1, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT},
      cub::SegmentedSortSubWarpMergeSortPolicy{
        BlockThreads, 4, 1, cub::WARP_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::WARP_STORE_DIRECT},
      1000000};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

CUB_TEST("DeviceSegmentedSort::SortPairs can be tuned", "[segmented_sort][pairs][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "DeviceSegmentedSort::SortPairsDescending can be tuned", "[segmented_sort][pairs][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

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

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedSort::StableSortPairs can be tuned", "[segmented_sort][pairs][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceSegmentedSort::StableSortPairsDescending can be tuned",
         "[segmented_sort][pairs][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys_in                             = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out                            = c2h::device_vector<int>(7);
  auto values_in                           = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out                          = c2h::device_vector<int>(7);
  auto begin_offsets                       = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);

  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));

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

  const c2h::device_vector<int> expected_keys{9, 8, 7, 6, 5, 3, 0};
  const c2h::device_vector<int> expected_values{6, 0, 2, 1, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

// The two-phase overloads exercised by the tests below. The keys of all test inputs are unique, so stable and unstable
// sorts produce the same result.
enum class two_phase_api
{
  sort_pairs,
  sort_pairs_descending,
  stable_sort_pairs,
  stable_sort_pairs_descending
};

bool is_descending(two_phase_api api)
{
  return api == two_phase_api::sort_pairs_descending || api == two_phase_api::stable_sort_pairs_descending;
}

// Calls the two-phase overload of api directly
template <class... Args>
cudaError_t call_two_phase(two_phase_api api, Args&&... args)
{
  switch (api)
  {
    case two_phase_api::sort_pairs:
      return cub::DeviceSegmentedSort::SortPairs(::cuda::std::forward<Args>(args)...);
    case two_phase_api::sort_pairs_descending:
      return cub::DeviceSegmentedSort::SortPairsDescending(::cuda::std::forward<Args>(args)...);
    case two_phase_api::stable_sort_pairs:
      return cub::DeviceSegmentedSort::StableSortPairs(::cuda::std::forward<Args>(args)...);
    case two_phase_api::stable_sort_pairs_descending:
      return cub::DeviceSegmentedSort::StableSortPairsDescending(::cuda::std::forward<Args>(args)...);
  }
  return cudaErrorInvalidValue;
}

// Calls the two-phase overload of api through its launch wrapper
template <class... Args>
void launch_two_phase(two_phase_api api, Args... args)
{
  switch (api)
  {
    case two_phase_api::sort_pairs:
      sort_pairs_two_phase(args...);
      break;
    case two_phase_api::sort_pairs_descending:
      sort_pairs_descending_two_phase(args...);
      break;
    case two_phase_api::stable_sort_pairs:
      stable_sort_pairs_two_phase(args...);
      break;
    case two_phase_api::stable_sort_pairs_descending:
      stable_sort_pairs_descending_two_phase(args...);
      break;
  }
}

// Runs the two-phase overload wrapped by two_phase: queries the temporary storage size, executes with the allocated
// storage, and checks that all kernels are launched on the stream of env.
template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const stream_scope scope{cuda::__call_or(cuda::get_stream, cuda::stream_ref{cudaStream_t{}}, env).get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// Expected results of sorting the keys {8, 6, 7, 5, 3, 0, 9} and values {0, 1, 2, 3, 4, 5, 6} in segments {0, 3, 7}
c2h::device_vector<int> expected_two_segment_keys(two_phase_api api)
{
  return is_descending(api)
         ? c2h::device_vector<int>{8, 7, 6, 9, 5, 3, 0}
         : c2h::device_vector<int>{6, 7, 8, 0, 3, 5, 9};
}

c2h::device_vector<int> expected_two_segment_values(two_phase_api api)
{
  return is_descending(api)
         ? c2h::device_vector<int>{0, 2, 1, 6, 3, 4, 5}
         : c2h::device_vector<int>{1, 2, 0, 5, 4, 3, 6};
}

#if TEST_LAUNCH == 0

// The launch wrappers copy their arguments, which the non-copyable stream types of test_with_custom_streams and the
// DoubleBuffer outputs do not allow, so these tests call the two-phase overloads directly. Test them with host launch
// only.

CUB_TEST_CASE("DeviceSegmentedSort pairs two-phase works with custom streams",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  const auto api = GENERATE(
    two_phase_api::sort_pairs,
    two_phase_api::sort_pairs_descending,
    two_phase_api::stable_sort_pairs,
    two_phase_api::stable_sort_pairs_descending);

  const auto offsets         = c2h::device_vector<int>{0, 3, 7};
  const int* d_offsets       = thrust::raw_pointer_cast(offsets.data());
  const auto expected_keys   = expected_two_segment_keys(api);
  const auto expected_values = expected_two_segment_values(api);

  SECTION("pointers")
  {
    const auto keys_in   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
    const auto values_in = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
    auto keys_out        = c2h::device_vector<int>(7, thrust::no_init);
    auto values_out      = c2h::device_vector<int>(7, thrust::no_init);

    test_with_custom_streams([&](const auto& env) {
      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return call_two_phase(
          api,
          d_temp_storage,
          temp_storage_bytes,
          thrust::raw_pointer_cast(keys_in.data()),
          thrust::raw_pointer_cast(keys_out.data()),
          thrust::raw_pointer_cast(values_in.data()),
          thrust::raw_pointer_cast(values_out.data()),
          7,
          2,
          d_offsets,
          d_offsets + 1,
          env);
      });

      REQUIRE(keys_out == expected_keys);
      REQUIRE(values_out == expected_values);
    });
  }

  SECTION("DoubleBuffer")
  {
    auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
    auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
    auto keys_buf1   = c2h::device_vector<int>(7, thrust::no_init);
    auto values_buf1 = c2h::device_vector<int>(7, thrust::no_init);

    test_with_custom_streams([&](const auto& env) {
      cub::DoubleBuffer<int> d_keys(
        thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
      cub::DoubleBuffer<int> d_values(
        thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

      run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
        return call_two_phase(
          api, d_temp_storage, temp_storage_bytes, d_keys, d_values, 7, 2, d_offsets, d_offsets + 1, env);
      });

      REQUIRE(c2h::device_vector<int>(d_keys.Current(), d_keys.Current() + 7) == expected_keys);
      REQUIRE(c2h::device_vector<int>(d_values.Current(), d_values.Current() + 7) == expected_values);
    });
  }
}

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr, a literal 0 or a null cudaStream_t for the stream exist. All must keep compiling and keep running on the
// default stream.
CUB_TEST_CASE("DeviceSegmentedSort pairs DoubleBuffer two-phase overloads accept legacy null stream arguments",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  const auto api = GENERATE(
    two_phase_api::sort_pairs,
    two_phase_api::sort_pairs_descending,
    two_phase_api::stable_sort_pairs,
    two_phase_api::stable_sort_pairs_descending);

  const auto offsets   = c2h::device_vector<int>{0, 3, 7};
  const int* d_offsets = thrust::raw_pointer_cast(offsets.data());
  auto keys_buf0       = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto values_buf0     = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto keys_buf1       = c2h::device_vector<int>(7, thrust::no_init);
  auto values_buf1     = c2h::device_vector<int>(7, thrust::no_init);

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

  SECTION("nullptr")
  {
    run_two_phase(cudaStream_t{}, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return call_two_phase(
        api, d_temp_storage, temp_storage_bytes, d_keys, d_values, 7, 2, d_offsets, d_offsets + 1, nullptr);
    });
  }

  SECTION("literal 0")
  {
    run_two_phase(cudaStream_t{}, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return call_two_phase(
        api, d_temp_storage, temp_storage_bytes, d_keys, d_values, 7, 2, d_offsets, d_offsets + 1, 0);
    });
  }

  SECTION("null cudaStream_t")
  {
    run_two_phase(cudaStream_t{}, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return call_two_phase(
        api, d_temp_storage, temp_storage_bytes, d_keys, d_values, 7, 2, d_offsets, d_offsets + 1, cudaStream_t{});
    });
  }

  REQUIRE(c2h::device_vector<int>(d_keys.Current(), d_keys.Current() + 7) == expected_two_segment_keys(api));
  REQUIRE(c2h::device_vector<int>(d_values.Current(), d_values.Current() + 7) == expected_two_segment_values(api));
}

#endif // TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceSegmentedSort pairs two-phase overloads accept legacy null stream arguments",
              "[segmented_sort][pairs][device]",
              CUB_SMALL)
{
  const auto api = GENERATE(
    two_phase_api::sort_pairs,
    two_phase_api::sort_pairs_descending,
    two_phase_api::stable_sort_pairs,
    two_phase_api::stable_sort_pairs_descending);

  const auto offsets   = c2h::device_vector<int>{0, 3, 7};
  const int* d_offsets = thrust::raw_pointer_cast(offsets.data());
  const auto keys_in   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  const auto values_in = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto keys_out        = c2h::device_vector<int>(7, thrust::no_init);
  auto values_out      = c2h::device_vector<int>(7, thrust::no_init);

  const int* d_keys_in   = thrust::raw_pointer_cast(keys_in.data());
  int* d_keys_out        = thrust::raw_pointer_cast(keys_out.data());
  const int* d_values_in = thrust::raw_pointer_cast(values_in.data());
  int* d_values_out      = thrust::raw_pointer_cast(values_out.data());

  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    launch_two_phase(api, d_keys_in, d_keys_out, d_values_in, d_values_out, 7, 2, d_offsets, d_offsets + 1, nullptr);
  }

  SECTION("literal 0")
  {
    launch_two_phase(api, d_keys_in, d_keys_out, d_values_in, d_values_out, 7, 2, d_offsets, d_offsets + 1, 0);
  }

  SECTION("null cudaStream_t")
  {
    launch_two_phase(
      api, d_keys_in, d_keys_out, d_values_in, d_values_out, 7, 2, d_offsets, d_offsets + 1, cudaStream_t{});
  }

  REQUIRE(keys_out == expected_two_segment_keys(api));
  REQUIRE(values_out == expected_two_segment_values(api));
}

#if TEST_LAUNCH != 1

CUB_TEST("DeviceSegmentedSort pairs two-phase can be tuned", "[segmented_sort][pairs][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto api = GENERATE(
    two_phase_api::sort_pairs,
    two_phase_api::sort_pairs_descending,
    two_phase_api::stable_sort_pairs,
    two_phase_api::stable_sort_pairs_descending);

  const auto begin_offsets = c2h::device_vector<int>{0};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_constant_iterator end_offsets(7, thrust::raw_pointer_cast(d_block_size.data()));
  const int* d_begin_offsets = thrust::raw_pointer_cast(begin_offsets.data());

  const auto expected_keys =
    is_descending(api) ? c2h::device_vector<int>{9, 8, 7, 6, 5, 3, 0} : c2h::device_vector<int>{0, 3, 5, 6, 7, 8, 9};
  const auto expected_values =
    is_descending(api) ? c2h::device_vector<int>{6, 0, 2, 1, 3, 4, 5} : c2h::device_vector<int>{5, 4, 3, 1, 2, 0, 6};

  const auto env = cuda::execution::tune(segmented_sort_tuning<target_block_size>{});

  SECTION("pointers")
  {
    const auto keys_in   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
    const auto values_in = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
    auto keys_out        = c2h::device_vector<int>(7, thrust::no_init);
    auto values_out      = c2h::device_vector<int>(7, thrust::no_init);

    launch_two_phase(
      api,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      7,
      1,
      d_begin_offsets,
      end_offsets,
      env);

    REQUIRE(keys_out == expected_keys);
    REQUIRE(values_out == expected_values);
  }

  // The launch wrappers copy their arguments, which loses the DoubleBuffer outputs, so call the overload directly
  SECTION("DoubleBuffer")
  {
    auto keys_buf0   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
    auto values_buf0 = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
    auto keys_buf1   = c2h::device_vector<int>(7, thrust::no_init);
    auto values_buf1 = c2h::device_vector<int>(7, thrust::no_init);

    cub::DoubleBuffer<int> d_keys(
      thrust::raw_pointer_cast(keys_buf0.data()), thrust::raw_pointer_cast(keys_buf1.data()));
    cub::DoubleBuffer<int> d_values(
      thrust::raw_pointer_cast(values_buf0.data()), thrust::raw_pointer_cast(values_buf1.data()));

    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return call_two_phase(
        api, d_temp_storage, temp_storage_bytes, d_keys, d_values, 7, 1, d_begin_offsets, end_offsets, env);
    });

    REQUIRE(c2h::device_vector<int>(d_keys.Current(), d_keys.Current() + 7) == expected_keys);
    REQUIRE(c2h::device_vector<int>(d_values.Current(), d_values.Current() + 7) == expected_values);
  }

  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

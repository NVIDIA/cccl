// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_radix_sort.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairsDescending, sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedRadixSort::SortKeysDescending, sort_keys_descending);

// %PARAM% TEST_LAUNCH lid 0:1

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceSegmentedRadixSort::SortPairs works with default environment", "[segmented_radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairs(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<int>(keys_in.size()),
      3,
      offsets.begin(),
      offsets.begin() + 1));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedRadixSort::SortPairsDescending works with default environment",
          "[segmented_radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairsDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<int>(keys_in.size()),
      3,
      offsets.begin(),
      offsets.begin() + 1));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

TEST_CASE("DeviceSegmentedRadixSort::SortKeys works with default environment", "[segmented_radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeys(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<int>(keys_in.size()),
      3,
      offsets.begin(),
      offsets.begin() + 1));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("DeviceSegmentedRadixSort::SortKeysDescending works with default environment",
          "[segmented_radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeysDescending(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<int>(keys_in.size()),
      3,
      offsets.begin(),
      offsets.begin() + 1));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("DeviceSegmentedRadixSort::SortKeys DoubleBuffer works with default environment",
          "[segmented_radix_sort][device]")
{
  auto keys_buf = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedRadixSort::SortKeys(
            d_keys, static_cast<int>(keys_buf.size()), 3, offsets.begin(), offsets.begin() + 1));

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(result_keys == expected_keys);
}

TEST_CASE("DeviceSegmentedRadixSort::SortKeysDescending DoubleBuffer works with default environment",
          "[segmented_radix_sort][device]")
{
  auto keys_buf = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedRadixSort::SortKeysDescending(
            d_keys, static_cast<int>(keys_buf.size()), 3, offsets.begin(), offsets.begin() + 1));

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(result_keys == expected_keys);
}

#endif

C2H_TEST("DeviceSegmentedRadixSort::SortPairs uses environment", "[segmented_radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairsDescending uses environment", "[segmented_radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeys uses environment", "[segmented_radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected_keys);
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeysDescending uses environment", "[segmented_radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected_keys);
}

TEST_CASE("DeviceSegmentedRadixSort::SortPairs uses custom stream", "[segmented_radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("DeviceSegmentedRadixSort::SortPairsDescending uses custom stream", "[segmented_radix_sort][device]")
{
  auto keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out   = c2h::device_vector<int>(7);
  auto values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_out = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(keys_out == expected_keys);
  REQUIRE(values_out == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("DeviceSegmentedRadixSort::SortKeys uses custom stream", "[segmented_radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(keys_out == expected_keys);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("DeviceSegmentedRadixSort::SortKeysDescending uses custom stream", "[segmented_radix_sort][device]")
{
  auto keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_out = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      static_cast<::cuda::std::int64_t>(keys_in.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    static_cast<int>(keys_in.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(keys_out == expected_keys);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeys DoubleBuffer uses environment", "[segmented_radix_sort][device]")
{
  auto keys_buf = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys(
    d_keys,
    static_cast<int>(keys_buf.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  REQUIRE(result_keys == expected_keys);
}

C2H_TEST("DeviceSegmentedRadixSort::SortKeysDescending DoubleBuffer uses environment", "[segmented_radix_sort][device]")
{
  auto keys_buf = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt = c2h::device_vector<int>(7);
  auto offsets  = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortKeysDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      static_cast<::cuda::std::int64_t>(keys_buf.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_keys_descending(
    d_keys,
    static_cast<int>(keys_buf.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  REQUIRE(result_keys == expected_keys);
}

TEST_CASE("DeviceSegmentedRadixSort::SortPairs DoubleBuffer works with default environment",
          "[segmented_radix_sort][device]")
{
  auto keys_buf   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt   = c2h::device_vector<int>(7);
  auto values_buf = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_alt = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf.data()), thrust::raw_pointer_cast(values_alt.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedRadixSort::SortPairs(
            d_keys, d_values, static_cast<int>(keys_buf.size()), 3, offsets.begin(), offsets.begin() + 1));

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);
  c2h::device_vector<int> result_values(
    thrust::device_pointer_cast(d_values.Current()), thrust::device_pointer_cast(d_values.Current()) + 7);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

TEST_CASE("DeviceSegmentedRadixSort::SortPairsDescending DoubleBuffer works with default environment",
          "[segmented_radix_sort][device]")
{
  auto keys_buf   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt   = c2h::device_vector<int>(7);
  auto values_buf = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_alt = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf.data()), thrust::raw_pointer_cast(values_alt.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_keys, d_values, static_cast<int>(keys_buf.size()), 3, offsets.begin(), offsets.begin() + 1));

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);
  c2h::device_vector<int> result_values(
    thrust::device_pointer_cast(d_values.Current()), thrust::device_pointer_cast(d_values.Current()) + 7);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairs DoubleBuffer uses environment", "[segmented_radix_sort][device]")
{
  auto keys_buf   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt   = c2h::device_vector<int>(7);
  auto values_buf = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_alt = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf.data()), thrust::raw_pointer_cast(values_alt.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(
    d_keys,
    d_values,
    static_cast<int>(keys_buf.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);
  c2h::device_vector<int> result_values(
    thrust::device_pointer_cast(d_values.Current()), thrust::device_pointer_cast(d_values.Current()) + 7);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

C2H_TEST("DeviceSegmentedRadixSort::SortPairsDescending DoubleBuffer uses environment",
         "[segmented_radix_sort][device]")
{
  auto keys_buf   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt   = c2h::device_vector<int>(7);
  auto values_buf = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_alt = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf.data()), thrust::raw_pointer_cast(values_alt.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  sort_pairs_descending(
    d_keys,
    d_values,
    static_cast<int>(keys_buf.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);
  c2h::device_vector<int> result_values(
    thrust::device_pointer_cast(d_values.Current()), thrust::device_pointer_cast(d_values.Current()) + 7);

  c2h::device_vector<int> expected_keys{8, 7, 6, 9, 5, 3, 0};
  c2h::device_vector<int> expected_values{0, 2, 1, 6, 3, 4, 5};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

TEST_CASE("DeviceSegmentedRadixSort::SortPairs DoubleBuffer uses custom stream", "[segmented_radix_sort][device]")
{
  auto keys_buf   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto keys_alt   = c2h::device_vector<int>(7);
  auto values_buf = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto values_alt = c2h::device_vector<int>(7);
  auto offsets    = c2h::device_vector<int>{0, 3, 3, 7};

  cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys_buf.data()), thrust::raw_pointer_cast(keys_alt.data()));
  cub::DoubleBuffer<int> d_values(
    thrust::raw_pointer_cast(values_buf.data()), thrust::raw_pointer_cast(values_alt.data()));

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys,
      d_values,
      static_cast<::cuda::std::int64_t>(keys_buf.size()),
      static_cast<::cuda::std::int64_t>(3),
      offsets.begin(),
      offsets.begin() + 1));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  sort_pairs(
    d_keys,
    d_values,
    static_cast<int>(keys_buf.size()),
    3,
    offsets.begin(),
    offsets.begin() + 1,
    0,
    static_cast<int>(sizeof(int) * 8),
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> result_keys(
    thrust::device_pointer_cast(d_keys.Current()), thrust::device_pointer_cast(d_keys.Current()) + 7);
  c2h::device_vector<int> result_values(
    thrust::device_pointer_cast(d_values.Current()), thrust::device_pointer_cast(d_values.Current()) + 7);

  c2h::device_vector<int> expected_keys{6, 7, 8, 0, 3, 5, 9};
  c2h::device_vector<int> expected_values{1, 2, 0, 5, 4, 3, 6};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairs, device_merge_sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeys, device_merge_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortPairs, device_merge_stable_sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, device_merge_stable_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairsCopy, device_merge_sort_pairs_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeysCopy, device_merge_sort_keys_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeysCopy, device_merge_stable_sort_keys_copy);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceMergeSort::SortPairs works with default environment", "[merge_sort][device]")
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::SortPairs(
            d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

TEST_CASE("DeviceMergeSort::SortKeys works with default environment", "[merge_sort][device]")
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

TEST_CASE("DeviceMergeSort::StableSortPairs works with default environment", "[merge_sort][device]")
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 6, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::StableSortPairs(
            d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

TEST_CASE("DeviceMergeSort::StableSortKeys works with default environment", "[merge_sort][device]")
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::StableSortKeys(
            d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

TEST_CASE("DeviceMergeSort::SortPairsCopy works with default environment", "[merge_sort][device]")
{
  auto d_keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out   = c2h::device_vector<int>(7);
  auto d_values_out = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortPairsCopy(
      d_keys_in.data().get(),
      d_values_in.data().get(),
      d_keys_out.data().get(),
      d_values_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

TEST_CASE("DeviceMergeSort::SortKeysCopy works with default environment", "[merge_sort][device]")
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeysCopy(
      d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

TEST_CASE("DeviceMergeSort::StableSortKeysCopy works with default environment", "[merge_sort][device]")
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeysCopy(
      d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

#endif

C2H_TEST("DeviceMergeSort::SortPairs uses environment", "[merge_sort][device]")
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys.data().get(),
      d_values.data().get(),
      static_cast<int>(d_keys.size()),
      cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_pairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

C2H_TEST("DeviceMergeSort::SortKeys uses environment", "[merge_sort][device]")
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeys(
      nullptr, expected_bytes_allocated, d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

C2H_TEST("DeviceMergeSort::StableSortPairs uses environment", "[merge_sort][device]")
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 6, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys.data().get(),
      d_values.data().get(),
      static_cast<int>(d_keys.size()),
      cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_stable_sort_pairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

C2H_TEST("DeviceMergeSort::StableSortKeys uses environment", "[merge_sort][device]")
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeys(
      nullptr, expected_bytes_allocated, d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_stable_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

TEST_CASE("DeviceMergeSort::SortPairs uses custom stream", "[merge_sort][device]")
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortPairs(
      nullptr,
      expected_bytes_allocated,
      d_keys.data().get(),
      d_values.data().get(),
      static_cast<int>(d_keys.size()),
      cuda::std::less<int>{}));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_pairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

TEST_CASE("DeviceMergeSort::StableSortKeys uses custom stream", "[merge_sort][device]")
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeys(
      nullptr, expected_bytes_allocated, d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_merge_stable_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

C2H_TEST("DeviceMergeSort::SortPairsCopy uses environment", "[merge_sort][device]")
{
  auto d_keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out   = c2h::device_vector<int>(7);
  auto d_values_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortPairsCopy(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.data().get(),
      d_values_in.data().get(),
      d_keys_out.data().get(),
      d_values_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_pairs_copy(
    d_keys_in.data().get(),
    d_values_in.data().get(),
    d_keys_out.data().get(),
    d_values_out.data().get(),
    static_cast<int>(d_keys_in.size()),
    cuda::std::less<int>{},
    env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

C2H_TEST("DeviceMergeSort::SortKeysCopy uses environment", "[merge_sort][device]")
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeysCopy(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.data().get(),
      d_keys_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}, env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

C2H_TEST("DeviceMergeSort::StableSortKeysCopy uses environment", "[merge_sort][device]")
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeysCopy(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.data().get(),
      d_keys_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_stable_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}, env);

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

TEST_CASE("DeviceMergeSort::SortKeysCopy uses custom stream", "[merge_sort][device]")
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeysCopy(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.data().get(),
      d_keys_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref, expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}, env);

  stream.sync();

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

TEST_CASE("DeviceMergeSort::StableSortKeysCopy uses custom stream", "[merge_sort][device]")
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7);

  cuda::stream stream{cuda::devices[0]};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeysCopy(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.data().get(),
      d_keys_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref, expected_allocation_size(expected_bytes_allocated)};

  device_merge_stable_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}, env);

  stream.sync();

  c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

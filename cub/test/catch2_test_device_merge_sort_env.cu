// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/execution>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_launch_helper.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::SortPairs, device_merge_sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::SortKeys, device_merge_sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::StableSortPairs, device_merge_stable_sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::StableSortKeys, device_merge_stable_sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::SortPairsCopy, device_merge_sort_pairs_copy);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::SortKeysCopy, device_merge_sort_keys_copy);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMergeSort::StableSortKeysCopy, device_merge_stable_sort_keys_copy);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

template <int ThreadsPerBlock>
struct merge_sort_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::MergeSortPolicy
  {
    return {ThreadsPerBlock, 1, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::BLOCK_STORE_DIRECT};
  }
};

#if TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceMergeSort::SortPairs works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::SortPairs(
            d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::SortKeys works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortPairs works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 6, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::StableSortPairs(
            d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortKeys works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::StableSortKeys(
            d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::SortPairsCopy works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out   = c2h::device_vector<int>(7, thrust::no_init);
  auto d_values_out = c2h::device_vector<int>(7, thrust::no_init);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortPairsCopy(
      d_keys_in.data().get(),
      d_values_in.data().get(),
      d_keys_out.data().get(),
      d_values_out.data().get(),
      static_cast<int>(d_keys_in.size()),
      cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::SortKeysCopy works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7, thrust::no_init);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeysCopy(
      d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortKeysCopy works with default environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7, thrust::no_init);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeysCopy(
      d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}));

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

#endif

CUB_TEST("DeviceMergeSort::SortPairs uses environment", "[merge_sort][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST("DeviceMergeSort::SortKeys uses environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeys(
      nullptr, expected_bytes_allocated, d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

CUB_TEST("DeviceMergeSort::StableSortPairs uses environment", "[merge_sort][device]", CUB_SMALL)
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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST("DeviceMergeSort::StableSortKeys uses environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeys(
      nullptr, expected_bytes_allocated, d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_merge_stable_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

#if TEST_LAUNCH != 1

CUB_TEST_CASE("DeviceMergeSort::SortPairs uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_sort_pairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::SortKeys uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortPairs uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys   = c2h::device_vector<int>{8, 6, 6, 5, 3, 0, 9};
  auto d_values = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_stable_sort_pairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortKeys uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_stable_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

#endif // TEST_LAUNCH != 1

CUB_TEST("DeviceMergeSort::SortPairsCopy uses environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out   = c2h::device_vector<int>(7, thrust::no_init);
  auto d_values_out = c2h::device_vector<int>(7, thrust::no_init);

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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST("DeviceMergeSort::SortKeysCopy uses environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7, thrust::no_init);

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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

CUB_TEST("DeviceMergeSort::StableSortKeysCopy uses environment", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7, thrust::no_init);

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

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

#if TEST_LAUNCH != 1

CUB_TEST_CASE("DeviceMergeSort::SortPairsCopy uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in    = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in  = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out   = c2h::device_vector<int>(7, thrust::no_init);
  auto d_values_out = c2h::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_sort_pairs_copy(
    d_keys_in.data().get(),
    d_values_in.data().get(),
    d_keys_out.data().get(),
    d_values_out.data().get(),
    static_cast<int>(d_keys_in.size()),
    cuda::std::less<int>{},
    env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::SortKeysCopy uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}, env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortKeysCopy uses custom stream", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys_in  = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out = c2h::device_vector<int>(7, thrust::no_init);

  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref stream_ref{stream};
  auto env = stdexec::env{stream_ref};

  device_merge_stable_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), cuda::std::less<int>{}, env);

  stream.sync();

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

#endif // TEST_LAUNCH != 1

using block_size_compare_t = block_size_extracting_op<cuda::std::less<>>;

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

CUB_TEST("DeviceMergeSort::SortPairs can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<int> d_values{0, 1, 2, 3};
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_sort_pairs(d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), compare_op, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::SortPairsCopy can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys_in{4, 1, 3, 2};
  c2h::device_vector<int> d_values_in{0, 1, 2, 3};
  c2h::device_vector<int> d_keys_out(4, thrust::no_init);
  c2h::device_vector<int> d_values_out(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_sort_pairs_copy(
    d_keys_in.data().get(),
    d_values_in.data().get(),
    d_keys_out.data().get(),
    d_values_out.data().get(),
    static_cast<int>(d_keys_in.size()),
    compare_op,
    env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::SortKeys can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), compare_op, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::SortKeysCopy can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys_in{4, 1, 3, 2};
  c2h::device_vector<int> d_keys_out(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), compare_op, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::StableSortPairs can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<int> d_values{0, 1, 2, 3};
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_stable_sort_pairs(
    d_keys.data().get(), d_values.data().get(), static_cast<int>(d_keys.size()), compare_op, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::StableSortKeys can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_stable_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), compare_op, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::StableSortKeysCopy can be tuned", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys_in{4, 1, 3, 2};
  c2h::device_vector<int> d_keys_out(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env        = cuda::execution::tune(merge_sort_tuning<target_block_size>{});

  device_merge_stable_sort_keys_copy(
    d_keys_in.data().get(), d_keys_out.data().get(), static_cast<int>(d_keys_in.size()), compare_op, env);
  REQUIRE(d_block_size[0] == target_block_size);
}

struct no_unroll_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::MergeSortPolicy
  {
    return {256, 7, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::BLOCK_STORE_DIRECT, false};
  }
};

CUB_TEST_CASE("DeviceMergeSort::SortKeys works with unroll disabled", "[merge_sort][device]", CUB_SMALL)
{
  auto d_keys = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto env    = cuda::execution::tune(no_unroll_tuning{});

  device_merge_sort_keys(d_keys.data().get(), static_cast<int>(d_keys.size()), cuda::std::less<int>{}, env);

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

#endif // TEST_LAUNCH != 1

#if TEST_LAUNCH == 0

// The two-phase overloads take the same environment as the single-phase ones but never allocate, so they do not go
// through the launch wrappers and would run identically in every TEST_LAUNCH variant. Test them with host launch only.

template <class TwoPhaseFn>
void test_two_phase_env_kinds(size_t expected_temp_storage_bytes, TwoPhaseFn two_phase)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref default_stream{cudaStream_t{}};

  auto run = [&](const auto& env, cuda::stream_ref expected_stream) {
    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes, env));
    REQUIRE(temp_storage_bytes == expected_temp_storage_bytes);

    c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
    {
      // fails if a kernel is launched on another stream
      const stream_scope scope{expected_stream.get()};
      REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, env));
    }
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    expected_stream.sync();
  };

  SECTION("default environment")
  {
    run(stdexec::env<>{}, default_stream);
  }

  SECTION("cudaStream_t")
  {
    run(stream.get(), cuda::stream_ref{stream});
  }

  SECTION("cuda::stream")
  {
    run(stream, cuda::stream_ref{stream});
  }

  SECTION("cuda::stream_ref")
  {
    run(cuda::stream_ref{stream}, cuda::stream_ref{stream});
  }

  SECTION("environment with stream")
  {
    run(stdexec::env{cuda::stream_ref{stream}}, cuda::stream_ref{stream});
  }

  SECTION("cuda::execution::gpu")
  {
    run(cuda::execution::gpu, default_stream);
  }

  SECTION("cuda::execution::gpu with stream")
  {
    run(cuda::execution::gpu.with(cuda::get_stream, cuda::stream_ref{stream}), cuda::stream_ref{stream});
  }
}

CUB_TEST_CASE("DeviceMergeSort::SortPairs works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys          = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values        = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  const auto num_items = static_cast<int>(d_keys.size());

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::SortPairs(
            nullptr, expected_bytes, d_keys.data().get(), d_values.data().get(), num_items, cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys.data().get(),
      d_values.data().get(),
      num_items,
      cuda::std::less<int>{},
      env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::SortKeys works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys          = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  const auto num_items = static_cast<int>(d_keys.size());

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeys(nullptr, expected_bytes, d_keys.data().get(), num_items, cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys.data().get(), num_items, cuda::std::less<int>{}, env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortPairs works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys          = c2h::device_vector<int>{8, 6, 6, 5, 3, 0, 9};
  auto d_values        = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  const auto num_items = static_cast<int>(d_keys.size());

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::StableSortPairs(
            nullptr, expected_bytes, d_keys.data().get(), d_values.data().get(), num_items, cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::StableSortPairs(
      d_temp_storage,
      temp_storage_bytes,
      d_keys.data().get(),
      d_values.data().get(),
      num_items,
      cuda::std::less<int>{},
      env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 6, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortKeys works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys          = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  const auto num_items = static_cast<int>(d_keys.size());

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceMergeSort::StableSortKeys(
            nullptr, expected_bytes, d_keys.data().get(), num_items, cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::StableSortKeys(
      d_temp_storage, temp_storage_bytes, d_keys.data().get(), num_items, cuda::std::less<int>{}, env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::SortPairsCopy works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys_in       = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_values_in     = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6};
  auto d_keys_out      = c2h::device_vector<int>(7, thrust::no_init);
  auto d_values_out    = c2h::device_vector<int>(7, thrust::no_init);
  const auto num_items = static_cast<int>(d_keys_in.size());

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortPairsCopy(
      nullptr,
      expected_bytes,
      d_keys_in.data().get(),
      d_values_in.data().get(),
      d_keys_out.data().get(),
      d_values_out.data().get(),
      num_items,
      cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::SortPairsCopy(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in.data().get(),
      d_values_in.data().get(),
      d_keys_out.data().get(),
      d_values_out.data().get(),
      num_items,
      cuda::std::less<int>{},
      env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  const c2h::device_vector<int> expected_values{5, 4, 3, 1, 2, 0, 6};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST_CASE("DeviceMergeSort::SortKeysCopy works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys_in       = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out      = c2h::device_vector<int>(7, thrust::no_init);
  const auto num_items = static_cast<int>(d_keys_in.size());

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::SortKeysCopy(
      nullptr, expected_bytes, d_keys_in.data().get(), d_keys_out.data().get(), num_items, cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::SortKeysCopy(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in.data().get(),
      d_keys_out.data().get(),
      num_items,
      cuda::std::less<int>{},
      env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

CUB_TEST_CASE("DeviceMergeSort::StableSortKeysCopy works with user provided memory and environment",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys_in       = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_keys_out      = c2h::device_vector<int>(7, thrust::no_init);
  const auto num_items = static_cast<int>(d_keys_in.size());

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMergeSort::StableSortKeysCopy(
      nullptr, expected_bytes, d_keys_in.data().get(), d_keys_out.data().get(), num_items, cuda::std::less<int>{}));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMergeSort::StableSortKeysCopy(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in.data().get(),
      d_keys_out.data().get(),
      num_items,
      cuda::std::less<int>{},
      env);
  });

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys_out == expected_keys);
}

// callers of the former `cudaStream_t stream = nullptr` parameter may pass nullptr or 0 explicitly
CUB_TEST_CASE("DeviceMergeSort::SortKeys two-phase overload accepts null stream arguments",
              "[merge_sort][device]",
              CUB_SMALL)
{
  auto d_keys          = c2h::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  const auto num_items = static_cast<int>(d_keys.size());

  auto sort_keys_on = [&](const auto& stream) {
    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess
            == cub::DeviceMergeSort::SortKeys(
              nullptr, temp_storage_bytes, d_keys.data().get(), num_items, cuda::std::less<int>{}, stream));

    c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
    {
      const stream_scope scope{cudaStream_t{}};
      REQUIRE(
        cudaSuccess
        == cub::DeviceMergeSort::SortKeys(
          thrust::raw_pointer_cast(temp_storage.data()),
          temp_storage_bytes,
          d_keys.data().get(),
          num_items,
          cuda::std::less<int>{},
          stream));
    }
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  };

  SECTION("nullptr")
  {
    sort_keys_on(nullptr);
  }

  SECTION("literal 0")
  {
    sort_keys_on(0);
  }

  const c2h::device_vector<int> expected_keys{0, 3, 5, 6, 7, 8, 9};
  REQUIRE(d_keys == expected_keys);
}

template <class TwoPhaseFn, class EnvT>
void run_two_phase(TwoPhaseFn two_phase, const EnvT& env)
{
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes, env));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, env));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

CUB_TEST(
  "DeviceMergeSort::SortPairs can be tuned with user provided memory", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<int> d_values{0, 1, 2, 3};
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys.data().get(),
        d_values.data().get(),
        static_cast<int>(d_keys.size()),
        compare_op,
        env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  const c2h::device_vector<int> expected_values{1, 3, 2, 0};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "DeviceMergeSort::SortKeys can be tuned with user provided memory", "[merge_sort][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_keys.data().get(), static_cast<int>(d_keys.size()), compare_op, env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::StableSortPairs can be tuned with user provided memory",
         "[merge_sort][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<int> d_values{0, 1, 2, 3};
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::StableSortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys.data().get(),
        d_values.data().get(),
        static_cast<int>(d_keys.size()),
        compare_op,
        env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  const c2h::device_vector<int> expected_values{1, 3, 2, 0};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_values == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::StableSortKeys can be tuned with user provided memory",
         "[merge_sort][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{4, 1, 3, 2};
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::StableSortKeys(
        d_temp_storage, temp_storage_bytes, d_keys.data().get(), static_cast<int>(d_keys.size()), compare_op, env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  REQUIRE(d_keys == expected_keys);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::SortPairsCopy can be tuned with user provided memory",
         "[merge_sort][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys_in{4, 1, 3, 2};
  c2h::device_vector<int> d_values_in{0, 1, 2, 3};
  c2h::device_vector<int> d_keys_out(4, thrust::no_init);
  c2h::device_vector<int> d_values_out(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::SortPairsCopy(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.data().get(),
        d_values_in.data().get(),
        d_keys_out.data().get(),
        d_values_out.data().get(),
        static_cast<int>(d_keys_in.size()),
        compare_op,
        env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  const c2h::device_vector<int> expected_values{1, 3, 2, 0};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::SortKeysCopy can be tuned with user provided memory",
         "[merge_sort][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys_in{4, 1, 3, 2};
  c2h::device_vector<int> d_keys_out(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::SortKeysCopy(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.data().get(),
        d_keys_out.data().get(),
        static_cast<int>(d_keys_in.size()),
        compare_op,
        env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMergeSort::StableSortKeysCopy can be tuned with user provided memory",
         "[merge_sort][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys_in{4, 1, 3, 2};
  c2h::device_vector<int> d_keys_out(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const auto compare_op = block_size_compare_t{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMergeSort::StableSortKeysCopy(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.data().get(),
        d_keys_out.data().get(),
        static_cast<int>(d_keys_in.size()),
        compare_op,
        env);
    },
    cuda::execution::tune(merge_sort_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{1, 2, 3, 4};
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH == 0

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test MergeSortPolicy properties", "[merge_sort][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::MergeSortPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::MergeSortPolicy>);

  // aggregate init
  constexpr auto p1 = cub::MergeSortPolicy{
    256,
    11,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_DEFAULT,
    cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::MergeSortPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 11,
    .load_algorithm    = cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    .load_modifier     = cub::CacheLoadModifier::LOAD_DEFAULT,
    .store_algorithm   = cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1)
          == "MergeSortPolicy { .threads_per_block = 256, .items_per_thread = 11"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .store_algorithm = BLOCK_STORE_DIRECT, .unroll = 1 }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)

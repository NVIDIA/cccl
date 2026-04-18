// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergeKeys, merge_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergePairs, merge_pairs);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

struct block_size_extracting_less_t
{
  int* ptr;

  __device__ bool operator()(const int& lhs, const int& rhs) const
  {
    if (threadIdx.x == 0)
    {
      atomicMin(ptr, blockDim.x);
    }
    return lhs < rhs;
  }
};

template <int BlockThreads>
struct merge_tuning
{
  _CCCL_API constexpr auto operator()(cuda::arch_id /*arch*/) const -> cub::detail::merge::merge_policy
  {
    return {BlockThreads, 1, cub::LOAD_DEFAULT, cub::BLOCK_STORE_WARP_TRANSPOSE, false};
  }
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 64>, cuda::std::integral_constant<int, 128>>;

#if TEST_LAUNCH == 0

TEST_CASE("DeviceMerge::MergeKeys works with default environment", "[merge][device]")
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      keys1.begin(), static_cast<int>(keys1.size()), keys2.begin(), static_cast<int>(keys2.size()), result.begin()));

  c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

TEST_CASE("DeviceMerge::MergePairs works with default environment", "[merge][device]")
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

#endif

C2H_TEST("DeviceMerge::MergeKeys can be tuned", "[merge][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  auto keys1                      = c2h::device_vector<int>{0, 2, 5};
  auto keys2                      = c2h::device_vector<int>{0, 3, 3, 4};
  auto result                     = c2h::device_vector<int>(7);
  auto d_block_size               = c2h::device_vector<int>(1, 2048);

  block_size_extracting_less_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::__tune(merge_tuning<target_block_size>{});

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin(),
      block_size_check,
      env));

  c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceMerge::MergePairs can be tuned", "[merge][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  auto keys1                      = c2h::device_vector<int>{0, 2, 5};
  auto values1                    = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2                      = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2                    = c2h::device_vector<char>{'A', 'B', 'C', 'D'};
  auto result_keys                = c2h::device_vector<int>(7);
  auto result_values              = c2h::device_vector<char>(7);
  auto d_block_size               = c2h::device_vector<int>(1, 2048);

  block_size_extracting_less_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::__tune(merge_tuning<target_block_size>{});

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin(),
      block_size_check,
      env));

  c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceMerge::MergeKeys uses environment", "[merge][device]")
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin()));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  merge_keys(keys1.begin(),
             static_cast<int>(keys1.size()),
             keys2.begin(),
             static_cast<int>(keys2.size()),
             result.begin(),
             cuda::std::less<>{},
             env);

  c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

TEST_CASE("DeviceMerge::MergeKeys uses custom stream", "[merge][device]")
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin()));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  merge_keys(keys1.begin(),
             static_cast<int>(keys1.size()),
             keys2.begin(),
             static_cast<int>(keys2.size()),
             result.begin(),
             cuda::std::less<>{},
             env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

C2H_TEST("DeviceMerge::MergePairs uses environment", "[merge][device]")
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  merge_pairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    cuda::std::less<>{},
    env);

  c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

TEST_CASE("DeviceMerge::MergePairs uses custom stream", "[merge][device]")
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  merge_pairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    cuda::std::less<>{},
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

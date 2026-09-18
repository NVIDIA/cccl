// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/thread/thread_sort.cuh>

#include <thrust/memory.h>

#include <cuda/std/bit>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/generators.h>
#include <c2h/vector.h>

enum class sort_algorithm
{
  stable,
  unstable
};

struct popcount_less
{
  template <typename T>
  _CCCL_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
  {
    return cuda::std::popcount(lhs) < cuda::std::popcount(rhs);
  }
};

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread, typename CompareOp>
_CCCL_DEVICE _CCCL_FORCEINLINE void
sort(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op)
{
  if constexpr (Algorithm == sort_algorithm::stable)
  {
    cub::StableOddEvenSort(keys, values, compare_op);
  }
  else
  {
    cub::UnstablePairwiseSort(keys, values, compare_op);
  }
}

template <sort_algorithm Algorithm, typename KeyT, int ItemsPerThread, typename CompareOp>
_CCCL_KERNEL_ATTRIBUTES void sort_keys_kernel(const KeyT* keys_in, KeyT* keys_out, CompareOp compare_op)
{
  const int thread_offset = ItemsPerThread * static_cast<int>(threadIdx.x);
  KeyT keys[ItemsPerThread];
  cub::NullType values[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys[item] = keys_in[thread_offset + item];
  }

  sort<Algorithm>(keys, values, compare_op);

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys_out[thread_offset + item] = keys[item];
  }
}

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread, typename CompareOp>
_CCCL_KERNEL_ATTRIBUTES void sort_pairs_kernel(
  const KeyT* keys_in, KeyT* keys_out, const ValueT* values_in, ValueT* values_out, CompareOp compare_op)
{
  const int thread_offset = ItemsPerThread * static_cast<int>(threadIdx.x);
  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys[item]   = keys_in[thread_offset + item];
    values[item] = values_in[thread_offset + item];
  }

  sort<Algorithm>(keys, values, compare_op);

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    keys_out[thread_offset + item]   = keys[item];
    values_out[thread_offset + item] = values[item];
  }
}

template <typename KeyT, typename ValueT>
std::vector<std::pair<KeyT, ValueT>>
make_pairs(const c2h::host_vector<KeyT>& keys, const c2h::host_vector<ValueT>& values)
{
  std::vector<std::pair<KeyT, ValueT>> pairs(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i)
  {
    pairs[i] = {keys[i], values[i]};
  }
  return pairs;
}

using key_types             = c2h::type_list<std::uint32_t, std::uint64_t>;
using value_types           = c2h::type_list<std::uint32_t, std::uint64_t>;
using items_per_thread_list = c2h::enum_type_list<int, 1, 2, 3, 4, 5, 7, 8, 11, 16>;
using algorithms            = c2h::enum_type_list<sort_algorithm, sort_algorithm::stable, sort_algorithm::unstable>;

CUB_TEST("Thread sort keys works", "[sort][thread]", CUB_SMALL, key_types, items_per_thread_list, algorithms)
{
  using key_t                    = c2h::get<0, TestType>;
  constexpr int items_per_thread = c2h::get<1, TestType>::value;
  constexpr auto algorithm       = c2h::get<2, TestType>::value;
  constexpr int threads          = 128;
  constexpr int num_items        = threads * items_per_thread;

  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(num_items);
  c2h::gen(C2H_SEED(3), keys_in);

  const c2h::host_vector<key_t> expected_input = keys_in;
  sort_keys_kernel<algorithm, key_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), popcount_less{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<key_t> actual = keys_out;
  for (int thread = 0; thread < threads; ++thread)
  {
    const auto begin          = thread * items_per_thread;
    const auto expected_begin = expected_input.begin() + begin;
    const auto actual_begin   = actual.begin() + begin;

    REQUIRE(std::is_sorted(actual_begin, actual_begin + items_per_thread, popcount_less{}));
    REQUIRE(std::is_permutation(expected_begin, expected_begin + items_per_thread, actual_begin));

    if constexpr (algorithm == sort_algorithm::stable)
    {
      std::vector<key_t> expected(expected_begin, expected_begin + items_per_thread);
      std::stable_sort(expected.begin(), expected.end(), popcount_less{});
      REQUIRE(std::equal(expected.begin(), expected.end(), actual_begin));
    }
  }
}

CUB_TEST(
  "Thread sort pairs works", "[sort][thread]", CUB_SMALL, key_types, value_types, items_per_thread_list, algorithms)
{
  using key_t                    = c2h::get<0, TestType>;
  using value_t                  = c2h::get<1, TestType>;
  constexpr int items_per_thread = c2h::get<2, TestType>::value;
  constexpr auto algorithm       = c2h::get<3, TestType>::value;
  constexpr int threads          = 128;
  constexpr int num_items        = threads * items_per_thread;

  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(num_items);
  c2h::gen(C2H_SEED(3), keys_in);

  c2h::host_vector<value_t> values_in(num_items);
  std::iota(values_in.begin(), values_in.end(), value_t{});
  c2h::device_vector<value_t> device_values_in = values_in;
  c2h::device_vector<value_t> values_out(num_items);

  const c2h::host_vector<key_t> host_keys_in = keys_in;
  const auto expected_input                  = make_pairs(host_keys_in, values_in);

  sort_pairs_kernel<algorithm, key_t, value_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(device_values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    popcount_less{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<key_t> host_keys_out     = keys_out;
  const c2h::host_vector<value_t> host_values_out = values_out;
  const auto actual                               = make_pairs(host_keys_out, host_values_out);
  const auto pair_less                            = [](const auto& lhs, const auto& rhs) {
    return popcount_less{}(lhs.first, rhs.first);
  };

  for (int thread = 0; thread < threads; ++thread)
  {
    const auto begin          = thread * items_per_thread;
    const auto expected_begin = expected_input.begin() + begin;
    const auto actual_begin   = actual.begin() + begin;

    REQUIRE(std::is_sorted(actual_begin, actual_begin + items_per_thread, pair_less));
    REQUIRE(std::is_permutation(expected_begin, expected_begin + items_per_thread, actual_begin));

    if constexpr (algorithm == sort_algorithm::stable)
    {
      std::vector<std::pair<key_t, value_t>> expected(expected_begin, expected_begin + items_per_thread);
      std::stable_sort(expected.begin(), expected.end(), pair_less);
      REQUIRE(std::equal(expected.begin(), expected.end(), actual_begin));
    }
  }
}

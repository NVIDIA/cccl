// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/thread/thread_sort.cuh>

#include <thrust/memory.h>

#include <cuda/std/bit>
#include <cuda/std/type_traits>

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

// use popcount to test sort stability
// example:
// {3, 5, 1}
// popcounts: 2, 2, 1
// stable sort must preserve the original order of equal elements:
// {1, 3, 5}
// unstable sort can change the order of equal elements:
// {1, 5, 3}
struct popcount_less
{
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    using unsigned_t  = cuda::std::make_unsigned_t<T>;
    auto lhs_unsigned = static_cast<unsigned_t>(lhs);
    auto rhs_unsigned = static_cast<unsigned_t>(rhs);
    return cuda::std::popcount(lhs_unsigned) < cuda::std::popcount(rhs_unsigned);
  }
};

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread, typename CompareOp>
__device__ void sort(KeyT (&keys)[ItemsPerThread], ValueT (&values)[ItemsPerThread], CompareOp compare_op)
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
__global__ void sort_keys_kernel(const KeyT* keys_in, KeyT* keys_out, CompareOp compare_op)
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
__global__ void sort_pairs_kernel(
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

using key_types             = c2h::type_list<uint32_t, int64_t>;
using value_types           = c2h::type_list<uint32_t, uint64_t>;
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
  c2h::host_vector<key_t> expected_input = keys_in;

  sort_keys_kernel<algorithm, key_t, items_per_thread><<<1, threads>>>(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), popcount_less{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<key_t> actual = keys_out;

  for (int thread = 0; thread < threads; ++thread)
  {
    auto begin          = thread * items_per_thread;
    auto expected_begin = expected_input.begin() + begin;
    auto actual_begin   = actual.begin() + begin;

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

//----------------------------------------------------------------------------------------------------------------------
// key-value sorting tests

template <typename KeyT, typename ValueT>
using pair_vector_t = std::vector<std::pair<KeyT, ValueT>>;

template <typename KeyT, typename ValueT>
pair_vector_t<KeyT, ValueT> make_pairs(const c2h::host_vector<KeyT>& keys, const c2h::host_vector<ValueT>& values)
{
  pair_vector_t<KeyT, ValueT> pairs(keys.size());
  for (size_t i = 0; i < keys.size(); ++i)
  {
    pairs[i] = {keys[i], values[i]};
  }
  return pairs;
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
  auto expected_input                        = make_pairs(host_keys_in, values_in);

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
  auto actual                                     = make_pairs(host_keys_out, host_values_out);
  auto pair_less                                  = [](auto lhs, auto rhs) {
    return popcount_less{}(lhs.first, rhs.first);
  };

  for (int thread = 0; thread < threads; ++thread)
  {
    auto begin          = thread * items_per_thread;
    auto expected_begin = expected_input.begin() + begin;
    auto actual_begin   = actual.begin() + begin;

    REQUIRE(std::is_sorted(actual_begin, actual_begin + items_per_thread, pair_less));
    REQUIRE(std::is_permutation(expected_begin, expected_begin + items_per_thread, actual_begin));

    if constexpr (algorithm == sort_algorithm::stable)
    {
      pair_vector_t<key_t, value_t> expected(expected_begin, expected_begin + items_per_thread);
      std::stable_sort(expected.begin(), expected.end(), pair_less);
      REQUIRE(std::equal(expected.begin(), expected.end(), actual_begin));
    }
  }
}

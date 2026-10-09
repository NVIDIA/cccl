// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Test of BlockMergeSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/block/block_merge_sort.cuh>

#include <thrust/sort.h>

#include <cuda/std/limits>

#include <algorithm>
#include <cstdint>
#include <ostream>

#include "cub_test_macros.h"

struct CustomLess
{
  template <typename DataType>
  __device__ __host__ bool operator()(const DataType& lhs, const DataType& rhs)
  {
    return lhs < rhs;
  }
};

//! A key type that deliberately has no `cuda::std::numeric_limits` specialization: no value of it is
//! ordered after every other value, so it can only be sorted through the overloads that take no
//! `oob_default`. This is what pins the contract of those overloads.
struct sentinel_free_key_t
{
  std::int32_t value;

  __host__ __device__ friend bool operator<(const sentinel_free_key_t& lhs, const sentinel_free_key_t& rhs)
  {
    return lhs.value < rhs.value;
  }

  __host__ __device__ friend bool operator==(const sentinel_free_key_t& lhs, const sentinel_free_key_t& rhs)
  {
    return lhs.value == rhs.value;
  }

  // Needed so that Catch2 can report the type on assertion failures.
  friend __host__ std::ostream& operator<<(std::ostream& os, const sentinel_free_key_t& self)
  {
    return os << self.value;
  }
};

static_assert(!::cuda::std::numeric_limits<sentinel_free_key_t>::is_specialized,
              "sentinel_free_key_t must stay without a numeric_limits specialization, otherwise it no longer covers "
              "key types that have no out-of-bounds default");

//! Out-of-bounds defaults are forwarded as a pack, so the sentinel-free instantiations neither accept
//! nor construct one.
template <int ThreadsInBlock, int ItemsPerThread, class KeyT, class ActionT, class... OobDefaultT>
__global__ void block_merge_sort_kernel(KeyT* data, int valid_items, ActionT action, OobDefaultT... oob_default)
{
  using BlockMergeSort = cub::BlockMergeSort<KeyT, ThreadsInBlock, ItemsPerThread>;

  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

  KeyT thread_data[ItemsPerThread];

  const int thread_offset = static_cast<int>(threadIdx.x) * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = idx < valid_items ? data[idx] : KeyT();
  }
  __syncthreads();

  BlockMergeSort sort(temp_storage_shuffle);

  action(sort, thread_data, valid_items, oob_default...);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;

    if (idx >= valid_items)
    {
      break;
    }

    data[idx] = thread_data[item];
  }
}

template <int ThreadsInBlock, int ItemsPerThread, class KeyT, class ValueT, class ActionT, class... OobDefaultT>
__global__ void
block_merge_sort_kernel(KeyT* keys, ValueT* vals, int valid_items, ActionT action, OobDefaultT... oob_default)
{
  using BlockMergeSort = cub::BlockMergeSort<KeyT, ThreadsInBlock, ItemsPerThread, ValueT>;

  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

  KeyT thread_keys[ItemsPerThread];
  ValueT thread_vals[ItemsPerThread];

  const int thread_offset = static_cast<int>(threadIdx.x) * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_keys[item] = idx < valid_items ? keys[idx] : KeyT{};
    thread_vals[item] = idx < valid_items ? vals[idx] : ValueT{};
  }
  __syncthreads();

  BlockMergeSort sort(temp_storage_shuffle);

  action(sort, thread_keys, thread_vals, valid_items, oob_default...);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;

    if (idx >= valid_items)
    {
      break;
    }

    keys[idx] = thread_keys[item];
    vals[idx] = thread_vals[item];
  }
}

struct stable_sort_keys_partial_tile_t
{
  static constexpr bool needs_sentinel = true;

  template <class BlockMergeSortT, class KeyT, class DefaultT>
  __device__ void operator()(BlockMergeSortT& sort, KeyT& thread_data, int valid_items, DefaultT oob_default) const
  {
    sort.StableSort(thread_data, CustomLess{}, valid_items, oob_default);
  }
};

// Counterpart of stable_sort_keys_partial_tile_t for the overload that takes no sentinel: it sorts
// the first valid_items keys without requiring a value that is ordered after all of them.
struct stable_sort_keys_partial_tile_no_sentinel_t
{
  static constexpr bool needs_sentinel = false;

  template <class BlockMergeSortT, class KeyT>
  __device__ void operator()(BlockMergeSortT& sort, KeyT& thread_data, int valid_items) const
  {
    sort.StableSort(thread_data, CustomLess{}, valid_items);
  }
};

struct stable_sort_pairs_partial_tile_t
{
  static constexpr bool needs_sentinel = true;

  template <class BlockMergeSortT, class KeyT, class ValueT, class DefaultT>
  __device__ void
  operator()(BlockMergeSortT& sort, KeyT& thread_keys, ValueT& thread_vals, int valid_items, DefaultT oob_default) const
  {
    sort.StableSort(thread_keys, thread_vals, CustomLess{}, valid_items, oob_default);
  }
};

struct stable_sort_pairs_partial_tile_no_sentinel_t
{
  static constexpr bool needs_sentinel = false;

  template <class BlockMergeSortT, class KeyT, class ValueT>
  __device__ void operator()(BlockMergeSortT& sort, KeyT& thread_keys, ValueT& thread_vals, int valid_items) const
  {
    sort.StableSort(thread_keys, thread_vals, CustomLess{}, valid_items);
  }
};

struct stable_sort_pairs_full_tile_t
{
  static constexpr bool needs_sentinel = false;

  template <class BlockMergeSortT, class KeyT, class ValueT>
  __device__ void operator()(BlockMergeSortT& sort, KeyT& thread_keys, ValueT& thread_vals, int /* valid_items */) const
  {
    sort.StableSort(thread_keys, thread_vals, CustomLess());
  }
};

struct stable_sort_keys_full_tile_t
{
  static constexpr bool needs_sentinel = false;

  template <class BlockMergeSortT, class KeyT>
  __device__ void operator()(BlockMergeSortT& sort, KeyT& thread_keys, int /* valid_items */) const
  {
    sort.StableSort(thread_keys, CustomLess());
  }
};

template <int ItemsPerThread, int ThreadsInBlock, class KeyT, class ActionT>
void block_merge_sort(c2h::device_vector<KeyT>& keys, ActionT action)
{
  const int valid_items = static_cast<int>(keys.size());

  if constexpr (ActionT::needs_sentinel)
  {
    block_merge_sort_kernel<ThreadsInBlock, ItemsPerThread><<<1, ThreadsInBlock>>>(
      thrust::raw_pointer_cast(keys.data()), valid_items, action, ::cuda::std::numeric_limits<KeyT>::max());
  }
  else
  {
    block_merge_sort_kernel<ThreadsInBlock, ItemsPerThread>
      <<<1, ThreadsInBlock>>>(thrust::raw_pointer_cast(keys.data()), valid_items, action);
  }

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int ItemsPerThread, int ThreadsInBlock, class KeyT, class ValueT, class ActionT>
void block_merge_sort(c2h::device_vector<KeyT>& keys, c2h::device_vector<ValueT>& vals, ActionT action)
{
  const int valid_items = static_cast<int>(keys.size());

  if constexpr (ActionT::needs_sentinel)
  {
    block_merge_sort_kernel<ThreadsInBlock, ItemsPerThread><<<1, ThreadsInBlock>>>(
      thrust::raw_pointer_cast(keys.data()),
      thrust::raw_pointer_cast(vals.data()),
      valid_items,
      action,
      ::cuda::std::numeric_limits<KeyT>::max());
  }
  else
  {
    block_merge_sort_kernel<ThreadsInBlock, ItemsPerThread><<<1, ThreadsInBlock>>>(
      thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(vals.data()), valid_items, action);
  }

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// %PARAM% THREADS_IN_BLOCK bs 64:256

using key_types        = c2h::type_list<std::int32_t, std::int64_t>;
using threads_in_block = c2h::enum_type_list<int, THREADS_IN_BLOCK>;
using items_per_thread = c2h::enum_type_list<int, 1, 2, 10, 15>;

// Both partial-tile interfaces have to produce the same sorted valid prefix
using partial_tile_key_actions =
  c2h::type_list<stable_sort_keys_partial_tile_t, stable_sort_keys_partial_tile_no_sentinel_t>;
using partial_tile_pair_actions =
  c2h::type_list<stable_sort_pairs_partial_tile_t, stable_sort_pairs_partial_tile_no_sentinel_t>;

template <class TestType>
struct params_t
{
  using key_t = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block = c2h::get<2, TestType>::value;
  static constexpr int tile_size        = items_per_thread * threads_in_block;
};

CUB_TEST("Block merge sort can sort keys in partial tiles",
         "[merge sort][block]",
         CUB_SMALL,
         key_types,
         items_per_thread,
         threads_in_block,
         partial_tile_key_actions)
{
  using params   = params_t<TestType>;
  using key_t    = typename params::key_t;
  using action_t = c2h::get<3, TestType>;

  c2h::device_vector<key_t> d_keys(GENERATE_COPY(take(10, random(0, params::tile_size))));

  c2h::gen(C2H_SEED(10), d_keys);

  c2h::host_vector<key_t> h_reference = d_keys;
  std::stable_sort(thrust::raw_pointer_cast(h_reference.data()),
                   thrust::raw_pointer_cast(h_reference.data()) + h_reference.size(),
                   CustomLess{});

  block_merge_sort<params::items_per_thread, params::threads_in_block>(d_keys, action_t{});

  REQUIRE(h_reference == d_keys);
}

CUB_TEST("Block merge sort can sort keys in full tiles",
         "[merge sort][block]",
         CUB_SMALL,
         key_types,
         items_per_thread,
         threads_in_block)
{
  using params = params_t<TestType>;
  using key_t  = typename params::key_t;

  c2h::device_vector<key_t> d_keys(params::tile_size);

  c2h::gen(C2H_SEED(10), d_keys);

  c2h::host_vector<key_t> h_reference = d_keys;
  std::stable_sort(thrust::raw_pointer_cast(h_reference.data()),
                   thrust::raw_pointer_cast(h_reference.data()) + h_reference.size(),
                   CustomLess{});

  block_merge_sort<params::items_per_thread, params::threads_in_block>(d_keys, stable_sort_keys_full_tile_t{});

  REQUIRE(h_reference == d_keys);
}

CUB_TEST("Block merge sort can sort pairs in partial tiles",
         "[merge sort][block]",
         CUB_SMALL,
         key_types,
         items_per_thread,
         threads_in_block,
         partial_tile_pair_actions)
{
  using params   = params_t<TestType>;
  using key_t    = typename params::key_t;
  using value_t  = key_t;
  using pair_t   = std::pair<key_t, value_t>;
  using action_t = c2h::get<3, TestType>;

  c2h::device_vector<key_t> d_keys(GENERATE_COPY(take(10, random(0, params::tile_size))));
  c2h::device_vector<value_t> d_vals(d_keys.size());

  c2h::gen(C2H_SEED(5), d_keys);
  c2h::gen(C2H_SEED(5), d_vals);

  c2h::host_vector<key_t> h_keys   = d_keys;
  c2h::host_vector<value_t> h_vals = d_vals;

  c2h::host_vector<pair_t> h_ref(d_keys.size());

  for (std::size_t idx = 0; idx < h_ref.size(); idx++)
  {
    h_ref[idx] = std::make_pair(h_keys[idx], h_vals[idx]);
  }

  std::stable_sort(thrust::raw_pointer_cast(h_ref.data()),
                   thrust::raw_pointer_cast(h_ref.data()) + h_ref.size(),
                   [](pair_t l, pair_t r) -> bool {
                     return l.first < r.first;
                   });

  for (std::size_t idx = 0; idx < h_ref.size(); idx++)
  {
    h_keys[idx] = h_ref[idx].first;
    h_vals[idx] = h_ref[idx].second;
  }

  block_merge_sort<params::items_per_thread, params::threads_in_block>(d_keys, d_vals, action_t{});

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_vals == d_vals);
}

// The two tests below use a key type that has no out-of-bounds default, so they only compile through
// the sentinel-free overloads. They are the coverage for the contract that those overloads require no
// value ordered after every valid key.
CUB_TEST("Block merge sort can sort keys of a type without a sentinel in partial tiles",
         "[merge sort][block]",
         CUB_SMALL,
         threads_in_block)
{
  constexpr int items_per_thread = 2;
  constexpr int threads_in_block = c2h::get<0, TestType>::value;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  const int num_items = GENERATE_COPY(take(4, random(0, tile_size)));

  c2h::device_vector<std::int32_t> d_raw(num_items);
  c2h::gen(C2H_SEED(4), d_raw);
  const c2h::host_vector<std::int32_t> h_raw = d_raw;

  c2h::host_vector<sentinel_free_key_t> h_keys(num_items);
  for (int idx = 0; idx < num_items; idx++)
  {
    h_keys[idx] = sentinel_free_key_t{h_raw[idx]};
  }
  c2h::device_vector<sentinel_free_key_t> d_keys = h_keys;

  c2h::host_vector<sentinel_free_key_t> h_reference = h_keys;
  std::stable_sort(thrust::raw_pointer_cast(h_reference.data()),
                   thrust::raw_pointer_cast(h_reference.data()) + h_reference.size(),
                   CustomLess{});

  block_merge_sort<items_per_thread, threads_in_block>(d_keys, stable_sort_keys_partial_tile_no_sentinel_t{});

  REQUIRE(h_reference == d_keys);
}

CUB_TEST("Block merge sort can sort pairs with a key type without a sentinel in partial tiles",
         "[merge sort][block]",
         CUB_SMALL,
         threads_in_block)
{
  using value_t = std::int32_t;
  using pair_t  = std::pair<sentinel_free_key_t, value_t>;

  constexpr int items_per_thread = 2;
  constexpr int threads_in_block = c2h::get<0, TestType>::value;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  const int num_items = GENERATE_COPY(take(4, random(0, tile_size)));

  c2h::device_vector<std::int32_t> d_raw(num_items);
  c2h::gen(C2H_SEED(4), d_raw);
  const c2h::host_vector<std::int32_t> h_raw = d_raw;

  c2h::host_vector<sentinel_free_key_t> h_keys(num_items);
  for (int idx = 0; idx < num_items; idx++)
  {
    h_keys[idx] = sentinel_free_key_t{h_raw[idx]};
  }
  c2h::device_vector<sentinel_free_key_t> d_keys = h_keys;

  c2h::device_vector<value_t> d_vals(num_items);
  c2h::gen(C2H_SEED(4), d_vals);
  c2h::host_vector<value_t> h_vals = d_vals;

  c2h::host_vector<pair_t> h_ref(num_items);
  for (int idx = 0; idx < num_items; idx++)
  {
    h_ref[idx] = std::make_pair(h_keys[idx], h_vals[idx]);
  }
  std::stable_sort(thrust::raw_pointer_cast(h_ref.data()),
                   thrust::raw_pointer_cast(h_ref.data()) + h_ref.size(),
                   [](pair_t l, pair_t r) -> bool {
                     return l.first < r.first;
                   });
  for (int idx = 0; idx < num_items; idx++)
  {
    h_keys[idx] = h_ref[idx].first;
    h_vals[idx] = h_ref[idx].second;
  }

  block_merge_sort<items_per_thread, threads_in_block>(d_keys, d_vals, stable_sort_pairs_partial_tile_no_sentinel_t{});

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_vals == d_vals);
}

CUB_TEST("Block merge sort can sort pairs in full tiles",
         "[merge sort][block]",
         CUB_SMALL,
         key_types,
         items_per_thread,
         threads_in_block)
{
  using params  = params_t<TestType>;
  using key_t   = typename params::key_t;
  using value_t = key_t;
  using pair_t  = std::pair<key_t, value_t>;

  c2h::device_vector<key_t> d_keys(params::tile_size);
  c2h::device_vector<value_t> d_vals(d_keys.size());

  c2h::gen(C2H_SEED(5), d_keys);
  c2h::gen(C2H_SEED(5), d_vals);

  c2h::host_vector<key_t> h_keys   = d_keys;
  c2h::host_vector<value_t> h_vals = d_vals;

  c2h::host_vector<pair_t> h_ref(d_keys.size());

  for (std::size_t idx = 0; idx < h_ref.size(); idx++)
  {
    h_ref[idx] = std::make_pair(h_keys[idx], h_vals[idx]);
  }

  std::stable_sort(thrust::raw_pointer_cast(h_ref.data()),
                   thrust::raw_pointer_cast(h_ref.data()) + h_ref.size(),
                   [](pair_t l, pair_t r) -> bool {
                     return l.first < r.first;
                   });

  for (std::size_t idx = 0; idx < h_ref.size(); idx++)
  {
    h_keys[idx] = h_ref[idx].first;
    h_vals[idx] = h_ref[idx].second;
  }

  block_merge_sort<params::items_per_thread, params::threads_in_block>(d_keys, d_vals, stable_sort_pairs_full_tile_t{});

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_vals == d_vals);
}

CUB_TEST("Block merge sort can sort pairs with mixed types", "[merge sort][block]", CUB_SMALL, threads_in_block)
{
  using key_t   = std::int32_t;
  using value_t = std::int64_t;
  using pair_t  = std::pair<key_t, value_t>;

  constexpr int items_per_thread = 2;
  constexpr int threads_in_block = c2h::get<0, TestType>::value;
  constexpr int tile_size        = items_per_thread * threads_in_block;

  c2h::device_vector<key_t> d_keys(tile_size);
  c2h::device_vector<value_t> d_vals(d_keys.size());

  c2h::gen(C2H_SEED(5), d_keys);
  c2h::gen(C2H_SEED(5), d_vals);

  c2h::host_vector<key_t> h_keys   = d_keys;
  c2h::host_vector<value_t> h_vals = d_vals;

  c2h::host_vector<pair_t> h_ref(d_keys.size());

  for (std::size_t idx = 0; idx < h_ref.size(); idx++)
  {
    h_ref[idx] = std::make_pair(h_keys[idx], h_vals[idx]);
  }

  std::stable_sort(thrust::raw_pointer_cast(h_ref.data()),
                   thrust::raw_pointer_cast(h_ref.data()) + h_ref.size(),
                   [](pair_t l, pair_t r) -> bool {
                     return l.first < r.first;
                   });

  for (std::size_t idx = 0; idx < h_ref.size(); idx++)
  {
    h_keys[idx] = h_ref[idx].first;
    h_vals[idx] = h_ref[idx].second;
  }

  block_merge_sort<items_per_thread, threads_in_block>(d_keys, d_vals, stable_sort_pairs_full_tile_t{});

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_vals == d_vals);
}

CUB_TEST("Block merge sort can sort large tiles", "[merge sort][block]", CUB_SMALL, threads_in_block)
{
  using key_t = std::uint16_t;

  constexpr int items_per_thread = 2;

  // Repurpose block sizes
  constexpr int cmake_threads_in_block = c2h::get<0, TestType>::value;
  constexpr int threads_in_block       = cmake_threads_in_block < 256 ? 512 : 1024;

  constexpr int tile_size = threads_in_block * items_per_thread;

  c2h::device_vector<key_t> d_keys(tile_size);
  c2h::gen(C2H_SEED(10), d_keys);

  c2h::host_vector<key_t> h_reference = d_keys;
  std::stable_sort(thrust::raw_pointer_cast(h_reference.data()),
                   thrust::raw_pointer_cast(h_reference.data()) + h_reference.size(),
                   CustomLess{});

  block_merge_sort<items_per_thread, threads_in_block>(d_keys, stable_sort_keys_full_tile_t{});

  REQUIRE(h_reference == d_keys);
}

CUB_TEST("Block merge sort is stable", "[merge sort][block]", CUB_SMALL, threads_in_block)
{
  using key_t = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  constexpr int items_per_thread = 2;
  constexpr int threads_in_block = c2h::get<0, TestType>::value;
  constexpr int tile_size        = threads_in_block * items_per_thread;

  c2h::device_vector<key_t> d_keys(tile_size);
  c2h::gen(C2H_SEED(10), d_keys);

  c2h::host_vector<key_t> h_reference = d_keys;
  std::stable_sort(thrust::raw_pointer_cast(h_reference.data()),
                   thrust::raw_pointer_cast(h_reference.data()) + h_reference.size(),
                   CustomLess{});

  block_merge_sort<items_per_thread, threads_in_block>(d_keys, stable_sort_keys_full_tile_t{});

  REQUIRE(h_reference == d_keys);
}

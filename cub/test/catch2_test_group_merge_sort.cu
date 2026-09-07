// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/detail/group_merge_sort.cuh>

#include <thrust/iterator/zip_iterator.h>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>

#include "cub_test_macros.h"
#include <c2h/custom_type.h>

struct CustomLess
{
  template <typename T>
  __device__ __host__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }
};

struct CustomGreater
{
  template <typename T>
  __device__ __host__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs > rhs;
  }
};

// -----------------------------------------------------------
// Action delegates for keys-only sorting
// -----------------------------------------------------------

struct group_sort_keys_full_tile_t
{
  template <typename KeyT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    int threads_per_group,
    int /*valid_items*/,
    KeyT /*oob_default*/) const
  {
    group_sort.Sort(thread_keys, CustomLess{}, threads_per_group);
  }
};

struct group_sort_keys_partial_tile_t
{
  template <typename KeyT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT oob_default) const
  {
    group_sort.Sort(thread_keys, CustomLess{}, threads_per_group, valid_items, oob_default);
  }
};

struct group_sort_keys_partial_tile_no_sentinel_t
{
  template <typename KeyT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT /*oob_default*/) const
  {
    group_sort.Sort(thread_keys, CustomLess{}, threads_per_group, valid_items);
  }
};

struct group_sort_keys_descending_t
{
  template <typename KeyT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT oob_default) const
  {
    group_sort.Sort(thread_keys, CustomGreater{}, threads_per_group, valid_items, oob_default);
  }
};

struct group_sort_call_helper_keys_t
{
  template <typename KeyT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* smem,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT oob_default) const
  {
    cub::detail::call_group_merge_runtime(
      smem,
      thread_keys,
      CustomLess{},
      threads_per_group,
      valid_items,
      oob_default,
      group_sort.get_member_tid(),
      group_sort.get_group_id());
  }
};

// -----------------------------------------------------------
// Action delegates for key-value pair sorting
// -----------------------------------------------------------

struct group_sort_pairs_full_tile_t
{
  template <typename KeyT, typename ValueT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*keys_smem*/,
    ValueT* /*values_smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    ValueT (&thread_values)[ITEMS_PER_THREAD],
    int threads_per_group,
    int /*valid_items*/,
    KeyT /*oob_default*/) const
  {
    group_sort.Sort(thread_keys, thread_values, CustomLess{}, threads_per_group);
  }
};

struct group_sort_pairs_partial_tile_t
{
  template <typename KeyT, typename ValueT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*keys_smem*/,
    ValueT* /*values_smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    ValueT (&thread_values)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT oob_default) const
  {
    group_sort.Sort(thread_keys, thread_values, CustomLess{}, threads_per_group, valid_items, oob_default);
  }
};

struct group_sort_pairs_partial_tile_no_sentinel_t
{
  template <typename KeyT, typename ValueT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*keys_smem*/,
    ValueT* /*values_smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    ValueT (&thread_values)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT /*oob_default*/) const
  {
    group_sort.Sort(thread_keys, thread_values, CustomLess{}, threads_per_group, valid_items);
  }
};

struct group_sort_pairs_descending_t
{
  template <typename KeyT, typename ValueT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* /*keys_smem*/,
    ValueT* /*values_smem*/,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    ValueT (&thread_values)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT oob_default) const
  {
    group_sort.Sort(thread_keys, thread_values, CustomGreater{}, threads_per_group, valid_items, oob_default);
  }
};

struct group_sort_call_helper_pairs_t
{
  template <typename KeyT, typename ValueT, typename GroupSortT, int ITEMS_PER_THREAD>
  __device__ void operator()(
    KeyT* keys_smem,
    ValueT* values_smem,
    GroupSortT& group_sort,
    KeyT (&thread_keys)[ITEMS_PER_THREAD],
    ValueT (&thread_values)[ITEMS_PER_THREAD],
    int threads_per_group,
    int valid_items,
    KeyT oob_default) const
  {
    cub::detail::call_group_merge_runtime(
      keys_smem,
      values_smem,
      thread_keys,
      thread_values,
      CustomLess{},
      threads_per_group,
      valid_items,
      oob_default,
      group_sort.get_member_tid(),
      group_sort.get_group_id());
  }
};

// -----------------------------------------------------------
// Test Kernels
// -----------------------------------------------------------

template <int ITEMS_PER_THREAD,
          int MAX_GROUP_THREADS,
          int BLOCK_THREADS,
          typename KeyT,
          typename SegmentSizeItT,
          typename ActionT>
__global__ void group_merge_sort_keys_kernel(
  const KeyT* d_in,
  KeyT* d_out,
  int threads_per_group,
  int total_segments,
  SegmentSizeItT segment_sizes,
  KeyT oob_default,
  ActionT action)
{
  static constexpr int CTA_SMEM_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD + BLOCK_THREADS;
  __shared__ KeyT storage_keys[CTA_SMEM_ITEMS];

  const int tid               = static_cast<int>(threadIdx.x);
  const int groups_per_block  = BLOCK_THREADS / threads_per_group;
  const int group_id_in_block = tid / threads_per_group;
  const int global_group_id   = static_cast<int>(blockIdx.x) * groups_per_block + group_id_in_block;

  if (global_group_id >= total_segments)
  {
    return;
  }

  const int member_tid        = tid % threads_per_group;
  const int group_smem_offset = group_id_in_block * (threads_per_group * ITEMS_PER_THREAD + 1);
  KeyT* group_keys_smem       = &storage_keys[group_smem_offset];

  using group_sort_t = cub::detail::GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS>;
  group_sort_t group_sort(
    group_keys_smem, static_cast<cub::NullType*>(nullptr), member_tid, group_id_in_block, threads_per_group);

  const int group_offset  = global_group_id * threads_per_group * ITEMS_PER_THREAD;
  const int thread_offset = group_offset + member_tid * ITEMS_PER_THREAD;
  const int valid_items   = segment_sizes[global_group_id];

  KeyT thread_keys[ITEMS_PER_THREAD];
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    thread_keys[item] = d_in[thread_offset + item];
  }

  action(group_keys_smem, group_sort, thread_keys, threads_per_group, valid_items, oob_default);

  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    const int item_idx          = member_tid * ITEMS_PER_THREAD + item;
    d_out[thread_offset + item] = (item_idx >= valid_items) ? oob_default : thread_keys[item];
  }
}

template <int ITEMS_PER_THREAD,
          int MAX_GROUP_THREADS,
          int BLOCK_THREADS,
          typename KeyT,
          typename ValueT,
          typename SegmentSizeItT,
          typename ActionT>
__global__ void group_merge_sort_pairs_kernel(
  const KeyT* d_keys_in,
  KeyT* d_keys_out,
  const ValueT* d_values_in,
  ValueT* d_values_out,
  int threads_per_group,
  int total_segments,
  SegmentSizeItT segment_sizes,
  KeyT oob_default,
  ActionT action)
{
  static constexpr int CTA_SMEM_ITEMS   = BLOCK_THREADS * ITEMS_PER_THREAD + BLOCK_THREADS;
  static constexpr int GROUP_ITEM_ALIGN = (alignof(KeyT) > alignof(ValueT)) ? alignof(KeyT) : alignof(ValueT);
  static constexpr int RAW_ITEM_SIZE    = (sizeof(KeyT) > sizeof(ValueT)) ? sizeof(KeyT) : sizeof(ValueT);
  static constexpr int GROUP_ITEM_SIZE = ((RAW_ITEM_SIZE + GROUP_ITEM_ALIGN - 1) / GROUP_ITEM_ALIGN) * GROUP_ITEM_ALIGN;
  __shared__ alignas(GROUP_ITEM_ALIGN) unsigned char cta_storage[CTA_SMEM_ITEMS * GROUP_ITEM_SIZE];

  const int tid               = static_cast<int>(threadIdx.x);
  const int groups_per_block  = BLOCK_THREADS / threads_per_group;
  const int group_id_in_block = tid / threads_per_group;
  const int global_group_id   = static_cast<int>(blockIdx.x) * groups_per_block + group_id_in_block;

  if (global_group_id >= total_segments)
  {
    return;
  }

  const int member_tid        = tid % threads_per_group;
  const int group_smem_offset = group_id_in_block * (threads_per_group * ITEMS_PER_THREAD + 1);
  unsigned char* group_smem   = cta_storage + group_smem_offset * GROUP_ITEM_SIZE;
  KeyT* group_keys_smem       = reinterpret_cast<KeyT*>(group_smem);
  ValueT* group_items_smem    = reinterpret_cast<ValueT*>(group_smem);

  using group_sort_t = cub::detail::GroupMergeSort<KeyT, ITEMS_PER_THREAD, MAX_GROUP_THREADS, ValueT>;
  group_sort_t group_sort(group_keys_smem, group_items_smem, member_tid, group_id_in_block, threads_per_group);

  const int group_offset  = global_group_id * threads_per_group * ITEMS_PER_THREAD;
  const int thread_offset = group_offset + member_tid * ITEMS_PER_THREAD;
  const int valid_items   = segment_sizes[global_group_id];

  KeyT thread_keys[ITEMS_PER_THREAD];
  ValueT thread_values[ITEMS_PER_THREAD];
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    thread_keys[item]   = d_keys_in[thread_offset + item];
    thread_values[item] = d_values_in[thread_offset + item];
  }

  action(group_keys_smem,
         group_items_smem,
         group_sort,
         thread_keys,
         thread_values,
         threads_per_group,
         valid_items,
         oob_default);

  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    const int item_idx                 = member_tid * ITEMS_PER_THREAD + item;
    d_keys_out[thread_offset + item]   = (item_idx >= valid_items) ? oob_default : thread_keys[item];
    d_values_out[thread_offset + item] = (item_idx >= valid_items) ? ValueT{} : thread_values[item];
  }
}

// Single-group TempStorage static-allocation validation kernel
template <int ITEMS_PER_THREAD, int GROUP_THREADS, typename KeyT>
__global__ void group_merge_sort_temp_storage_kernel(KeyT* d_data, int valid_items)
{
  using group_sort_t = cub::detail::GroupMergeSort<KeyT, ITEMS_PER_THREAD, GROUP_THREADS>;
  __shared__ typename group_sort_t::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);
  group_sort_t sort(temp_storage, tid, 0, GROUP_THREADS);

  KeyT keys[ITEMS_PER_THREAD];
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    keys[i] = d_data[tid * ITEMS_PER_THREAD + i];
  }

  sort.Sort(keys, CustomLess{}, GROUP_THREADS, valid_items);

  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    d_data[tid * ITEMS_PER_THREAD + i] = keys[i];
  }
}

// -----------------------------------------------------------
// Host reference computation
// -----------------------------------------------------------

template <typename RandomItT, typename SegmentSizeItT, typename KeyT, typename CompareOp = CustomLess>
void compute_host_group_reference(
  RandomItT h_data,
  SegmentSizeItT segment_sizes,
  unsigned int num_segments,
  KeyT oob_default,
  int group_tile_items,
  CompareOp comp = CustomLess{})
{
  for (unsigned int seg = 0; seg < num_segments; ++seg)
  {
    const unsigned int valid_count = static_cast<unsigned int>(segment_sizes[seg]);
    std::stable_sort(h_data, h_data + valid_count, comp);
    std::fill(h_data + valid_count, h_data + group_tile_items, oob_default);
    h_data += group_tile_items;
  }
}

template <typename ZipItT, typename SegmentSizeItT, typename KeyT, typename ValueT, typename CompareOp = CustomLess>
void compute_host_group_pairs_reference(
  ZipItT h_pairs,
  SegmentSizeItT segment_sizes,
  unsigned int num_segments,
  KeyT oob_default,
  ValueT val_default,
  int group_tile_items,
  CompareOp comp = CustomLess{})
{
  for (unsigned int seg = 0; seg < num_segments; ++seg)
  {
    const unsigned int valid_count = static_cast<unsigned int>(segment_sizes[seg]);
    std::stable_sort(h_pairs, h_pairs + valid_count, [comp](const auto& lhs, const auto& rhs) {
      return comp(thrust::get<0>(lhs), thrust::get<0>(rhs));
    });
    for (unsigned int i = valid_count; i < static_cast<unsigned int>(group_tile_items); ++i)
    {
      h_pairs[i] = thrust::make_tuple(oob_default, val_default);
    }
    h_pairs += group_tile_items;
  }
}

// -----------------------------------------------------------
// Type Lists for Catch2
// -----------------------------------------------------------

using custom_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;
using key_types   = c2h::type_list<std::uint8_t, std::int32_t, std::int64_t, custom_t>;
using value_types = c2h::type_list<std::int32_t, custom_t>;

// Group widths covering: sub-warp (2, 4, 16), warp (32), and multi-warp (64, 128)
using group_widths_list = c2h::enum_type_list<int, 2, 4, 16, 32, 64, 128>;

// Items per thread
using items_per_thread_list = c2h::enum_type_list<int, 1, 4, 7>;

template <typename TestType>
struct group_params_t
{
  using type                          = typename c2h::get<0, TestType>;
  static constexpr int group_width    = c2h::get<1, TestType>::value;
  static constexpr int items_per_th   = c2h::get<2, TestType>::value;
  static constexpr int block_size     = 256;
  static constexpr int max_group_th   = 256;
  static constexpr int groups_per_blk = block_size / group_width;
  static constexpr int num_blocks     = 2;
  static constexpr int total_groups   = groups_per_blk * num_blocks;
  static constexpr int items_per_grp  = group_width * items_per_th;
  static constexpr int total_items    = total_groups * items_per_grp;
};

// -----------------------------------------------------------
// Test Cases
// -----------------------------------------------------------

CUB_TEST("GroupMergeSort: Keys-only full tile sort works across sub-warp, warp, and multi-warp",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;

  c2h::device_vector<KeyT> d_in(params::total_items);
  c2h::device_vector<KeyT> d_out(params::total_items);
  auto segment_sizes     = cuda::constant_iterator(params::items_per_grp);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_in);

  group_merge_sort_keys_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      params::group_width,
      params::total_groups,
      segment_sizes,
      oob_default,
      group_sort_keys_full_tile_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_expected = d_in;
  compute_host_group_reference(
    h_expected.begin(), segment_sizes, params::total_groups, oob_default, params::items_per_grp);

  REQUIRE(h_expected == d_out);
}

CUB_TEST("GroupMergeSort: Keys-only partial tile sort works with runtime valid_items",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;

  c2h::device_vector<KeyT> d_in(params::total_items);
  c2h::device_vector<KeyT> d_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_keys_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_keys_partial_tile_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_expected   = d_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  compute_host_group_reference(
    h_expected.begin(), segment_sizes, params::total_groups, oob_default, params::items_per_grp);

  REQUIRE(h_expected == d_out);
}

CUB_TEST("GroupMergeSort: Keys-only partial tile sort without sentinel",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;

  c2h::device_vector<KeyT> d_in(params::total_items);
  c2h::device_vector<KeyT> d_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_keys_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_keys_partial_tile_no_sentinel_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_expected   = d_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  compute_host_group_reference(
    h_expected.begin(), segment_sizes, params::total_groups, oob_default, params::items_per_grp);

  REQUIRE(h_expected == d_out);
}

CUB_TEST("GroupMergeSort: Descending order sorting works",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;

  c2h::device_vector<KeyT> d_in(params::total_items);
  c2h::device_vector<KeyT> d_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::lowest();

  c2h::gen(C2H_SEED(2), d_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_keys_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_keys_descending_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_expected   = d_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  compute_host_group_reference(
    h_expected.begin(), segment_sizes, params::total_groups, oob_default, params::items_per_grp, CustomGreater{});

  REQUIRE(h_expected == d_out);
}

CUB_TEST("GroupMergeSort: Duplicate keys sorting works",
         "[sort][group]",
         CUB_SMALL,
         c2h::type_list<std::int32_t>,
         group_widths_list,
         items_per_thread_list)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;

  c2h::device_vector<KeyT> d_in(params::total_items);
  c2h::device_vector<KeyT> d_out(params::total_items);
  auto segment_sizes     = cuda::constant_iterator(params::items_per_grp);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  // Generate input with lots of duplicates in [0, 5]
  c2h::gen(C2H_SEED(2), d_in, 0, 5);

  group_merge_sort_keys_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      params::group_width,
      params::total_groups,
      segment_sizes,
      oob_default,
      group_sort_keys_full_tile_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_expected = d_in;
  compute_host_group_reference(
    h_expected.begin(), segment_sizes, params::total_groups, oob_default, params::items_per_grp);

  REQUIRE(h_expected == d_out);
}

CUB_TEST("GroupMergeSort: Key-value pair full tile sort works across sub-warp, warp, and multi-warp",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list,
         value_types)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;
  using ValueT = typename c2h::get<3, TestType>;

  c2h::device_vector<KeyT> d_keys_in(params::total_items);
  c2h::device_vector<KeyT> d_keys_out(params::total_items);
  c2h::device_vector<ValueT> d_values_in(params::total_items);
  c2h::device_vector<ValueT> d_values_out(params::total_items);
  auto segment_sizes     = cuda::constant_iterator(params::items_per_grp);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);

  group_merge_sort_pairs_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      params::group_width,
      params::total_groups,
      segment_sizes,
      oob_default,
      group_sort_pairs_full_tile_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_keys     = d_keys_in;
  c2h::host_vector<ValueT> h_values = d_values_in;
  auto h_pairs                      = thrust::make_zip_iterator(h_keys.begin(), h_values.begin());

  compute_host_group_pairs_reference(
    h_pairs, segment_sizes, params::total_groups, oob_default, ValueT{}, params::items_per_grp);

  REQUIRE(h_keys == d_keys_out);
  REQUIRE(h_values == d_values_out);
}

CUB_TEST("GroupMergeSort: Key-value pair sort works across sub-warp, warp, and multi-warp",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list,
         value_types)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;
  using ValueT = typename c2h::get<3, TestType>;

  c2h::device_vector<KeyT> d_keys_in(params::total_items);
  c2h::device_vector<KeyT> d_keys_out(params::total_items);
  c2h::device_vector<ValueT> d_values_in(params::total_items);
  c2h::device_vector<ValueT> d_values_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_pairs_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_pairs_partial_tile_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_keys       = d_keys_in;
  c2h::host_vector<ValueT> h_values   = d_values_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  auto h_pairs                        = thrust::make_zip_iterator(h_keys.begin(), h_values.begin());

  compute_host_group_pairs_reference(
    h_pairs, segment_sizes, params::total_groups, oob_default, ValueT{}, params::items_per_grp);

  REQUIRE(h_keys == d_keys_out);
  REQUIRE(h_values == d_values_out);
}

CUB_TEST("GroupMergeSort: Key-value pair partial tile sort without sentinel",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list,
         value_types)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;
  using ValueT = typename c2h::get<3, TestType>;

  c2h::device_vector<KeyT> d_keys_in(params::total_items);
  c2h::device_vector<KeyT> d_keys_out(params::total_items);
  c2h::device_vector<ValueT> d_values_in(params::total_items);
  c2h::device_vector<ValueT> d_values_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_pairs_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_pairs_partial_tile_no_sentinel_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_keys       = d_keys_in;
  c2h::host_vector<ValueT> h_values   = d_values_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  auto h_pairs                        = thrust::make_zip_iterator(h_keys.begin(), h_values.begin());

  compute_host_group_pairs_reference(
    h_pairs, segment_sizes, params::total_groups, oob_default, ValueT{}, params::items_per_grp);

  REQUIRE(h_keys == d_keys_out);
  REQUIRE(h_values == d_values_out);
}

CUB_TEST("GroupMergeSort: Key-value pair descending order sorting works",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list,
         value_types)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;
  using ValueT = typename c2h::get<3, TestType>;

  c2h::device_vector<KeyT> d_keys_in(params::total_items);
  c2h::device_vector<KeyT> d_keys_out(params::total_items);
  c2h::device_vector<ValueT> d_values_in(params::total_items);
  c2h::device_vector<ValueT> d_values_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::lowest();

  c2h::gen(C2H_SEED(2), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_pairs_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_pairs_descending_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_keys       = d_keys_in;
  c2h::host_vector<ValueT> h_values   = d_values_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  auto h_pairs                        = thrust::make_zip_iterator(h_keys.begin(), h_values.begin());

  compute_host_group_pairs_reference(
    h_pairs, segment_sizes, params::total_groups, oob_default, ValueT{}, params::items_per_grp, CustomGreater{});

  REQUIRE(h_keys == d_keys_out);
  REQUIRE(h_values == d_values_out);
}

CUB_TEST("GroupMergeSort: Key-value pair call_group_merge_runtime helper works",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list,
         value_types)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;
  using ValueT = typename c2h::get<3, TestType>;

  c2h::device_vector<KeyT> d_keys_in(params::total_items);
  c2h::device_vector<KeyT> d_keys_out(params::total_items);
  c2h::device_vector<ValueT> d_values_in(params::total_items);
  c2h::device_vector<ValueT> d_values_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_keys_in);
  c2h::gen(C2H_SEED(1), d_values_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_pairs_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_keys_in.data()),
      thrust::raw_pointer_cast(d_keys_out.data()),
      thrust::raw_pointer_cast(d_values_in.data()),
      thrust::raw_pointer_cast(d_values_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_call_helper_pairs_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_keys       = d_keys_in;
  c2h::host_vector<ValueT> h_values   = d_values_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  auto h_pairs                        = thrust::make_zip_iterator(h_keys.begin(), h_values.begin());

  compute_host_group_pairs_reference(
    h_pairs, segment_sizes, params::total_groups, oob_default, ValueT{}, params::items_per_grp);

  REQUIRE(h_keys == d_keys_out);
  REQUIRE(h_values == d_values_out);
}

CUB_TEST("GroupMergeSort: call_group_merge_runtime helper works",
         "[sort][group]",
         CUB_SMALL,
         key_types,
         group_widths_list,
         items_per_thread_list)
{
  using params = group_params_t<TestType>;
  using KeyT   = typename params::type;

  c2h::device_vector<KeyT> d_in(params::total_items);
  c2h::device_vector<KeyT> d_out(params::total_items);
  c2h::device_vector<int> d_segment_sizes(params::total_groups);
  const auto oob_default = cuda::std::numeric_limits<KeyT>::max();

  c2h::gen(C2H_SEED(2), d_in);
  c2h::gen(C2H_SEED(1), d_segment_sizes, 0, params::items_per_grp);

  group_merge_sort_keys_kernel<params::items_per_th, params::max_group_th, params::block_size>
    <<<params::num_blocks, params::block_size>>>(
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      params::group_width,
      params::total_groups,
      d_segment_sizes.cbegin(),
      oob_default,
      group_sort_call_helper_keys_t{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_expected   = d_in;
  c2h::host_vector<int> segment_sizes = d_segment_sizes;
  compute_host_group_reference(
    h_expected.begin(), segment_sizes, params::total_groups, oob_default, params::items_per_grp);

  REQUIRE(h_expected == d_out);
}

CUB_TEST("GroupMergeSort: TempStorage single group works", "[sort][group]", CUB_SMALL, key_types)
{
  using KeyT                     = typename c2h::get<0, TestType>;
  constexpr int GROUP_THREADS    = 64;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int TILE_ITEMS       = GROUP_THREADS * ITEMS_PER_THREAD;
  const int valid_items          = GENERATE(0, 1, 180, TILE_ITEMS);

  c2h::device_vector<KeyT> d_data(TILE_ITEMS);
  c2h::gen(C2H_SEED(2), d_data);
  c2h::host_vector<KeyT> h_expected = d_data;

  group_merge_sort_temp_storage_kernel<ITEMS_PER_THREAD, GROUP_THREADS>
    <<<1, GROUP_THREADS>>>(thrust::raw_pointer_cast(d_data.data()), valid_items);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  std::stable_sort(h_expected.begin(), h_expected.begin() + valid_items, CustomLess{});

  // The partial-tile sort without sentinel only guarantees that the first valid_items
  // elements are sorted in place; positions >= valid_items retain unspecified tile values.
  c2h::host_vector<KeyT> h_result = d_data;
  h_expected.resize(valid_items);
  h_result.resize(valid_items);

  REQUIRE(h_expected == h_result);
}

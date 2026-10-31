// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Expand GroupMerge settings into capacity bins and implement the cooperative device sorting worker.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_merge_sort.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/segmented_sort_entries.cuh>
#include <cub/thread/thread_sort.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/functional>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Enumerate the supported thread/IPT pairs and their capacities in strictly increasing capacity order.
template <typename PolicyGetter>
struct group_merge_expansion
{
  inline static constexpr auto policy = PolicyGetter{}();
  static_assert(policy.min_threads > 0 && (policy.min_threads & (policy.min_threads - 1)) == 0);
  static_assert(policy.max_threads >= policy.min_threads && (policy.max_threads & (policy.max_threads - 1)) == 0
                && policy.max_threads <= policy.block_threads);
  static_assert(policy.min_ipt > 0 && policy.max_ipt >= policy.min_ipt);
  static constexpr int ipt_choices = policy.max_ipt - policy.min_ipt + 1;
  // Count the power-of-two thread choices within the policy range.
  _CCCL_HOST_DEVICE_API static constexpr int thread_choices()
  {
    int count = 0;
    for (int threads = policy.min_threads; threads <= policy.max_threads; threads *= 2)
    {
      ++count;
    }
    return count;
  }
  static_assert(thread_choices() == 1 || policy.max_ipt < 2 * policy.min_ipt,
                "GroupMerge policy: with multiple thread counts, max_ipt must be < 2 * min_ipt so that threads * ipt "
                "strictly increases across bins (bin order must equal capacity order)");
  static constexpr int size = thread_choices() * ipt_choices;
  // Map a local bin to its thread/IPT pair during compile-time expansion.
  _CCCL_HOST_DEVICE_API static constexpr bin_config enumerate_shape(int bin_idx)
  {
    // Enumerate supported pairs at compile time, without encoding bin positions.
    for (int threads = policy.min_threads; threads <= policy.max_threads; threads *= 2)
    {
      for (int ipt = policy.min_ipt; ipt <= policy.max_ipt; ++ipt)
      {
        if (bin_idx-- == 0)
        {
          return {threads, ipt};
        }
      }
    }
    _CCCL_UNREACHABLE();
  }
  // Select each pair once and reuse it for its capacity and configuration type.
  template <int Bin>
  struct bin_selection
  {
    inline static constexpr bin_config chosen = enumerate_shape(Bin);
    static constexpr int capacity             = chosen.threads_per_segment * chosen.items_per_thread;
    using type                                = bin_shape<chosen.threads_per_segment, chosen.items_per_thread>;
  };
  // Build the capacity-list type from the selected configurations.
  template <::cuda::std::size_t... BinIndices>
  static auto expand(::cuda::std::index_sequence<BinIndices...>)
    -> capacity_list<bin_selection<BinIndices>::capacity...>;
  using capacities = decltype(expand(::cuda::std::make_index_sequence<size>{}));
  template <int Bin>
  using shape_for_bin = typename bin_selection<Bin>::type;
};

// Sort segments cooperatively using thread-local sorting and merges across threads.
// Warp workers have independent shared key storage and never use CTA barriers.
template <typename PolicyGetter>
struct group_merge_implementation
{
  inline static constexpr auto policy     = PolicyGetter{}();
  using expansion                         = group_merge_expansion<PolicyGetter>;
  static constexpr bool is_warp           = policy.max_threads <= 32;
  static constexpr int threads_per_worker = is_warp ? 32 : policy.block_threads;
  static_assert(policy.block_threads > 0 && policy.block_threads <= 1024);
  static_assert(policy.blocks_per_sm > 0 && policy.min_blocks_per_sm > 0);
  static_assert(policy.block_threads % 32 == 0);
  static_assert(is_warp || policy.min_threads >= 64);
  static_assert(policy.block_threads % policy.max_threads == 0);
  static_assert(policy.min_ipt >= 9 && policy.max_ipt <= 15, "GroupMerge workers support IPT 9..15");

  // Dynamic shared memory for key exchange/merging, including one prefetch slot per work unit.
  template <typename Key>
  _CCCL_HOST_DEVICE_API static constexpr size_t scratch_bytes()
  {
    return sizeof(Key) * (policy.block_threads * policy.max_ipt + (is_warp ? policy.block_threads / 32 : 1));
  }
  // Synchronize only this worker's warp or block so warp units can progress independently.
  _CCCL_DEVICE_API static void sync()
  {
    if constexpr (is_warp)
    {
      __syncwarp();
    }
    else
    {
      __syncthreads();
    }
  }

  // Obtain an input segment ID from segment_idx, load the segment, sort thread-local runs,
  // merge across cooperating threads, and write the valid keys back.
  // A segment_idx of ~0u masks an unused group while keeping its threads in synchronization.
  template <int ItemsPerThread, typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void sort_segment(
    unsigned segment_idx,
    bin_config config,
    const partition_result& bins,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    Key* keys_shared)
  {
    const ::cuda::std::less<Key> compare{};
    const int worker_thread_idx = threadIdx.x % threads_per_worker;
    if constexpr (is_warp)
    {
      keys_shared += (threadIdx.x / 32) * (32 * policy.max_ipt + 1);
    }
    const int thread_idx_in_group = worker_thread_idx % config.threads_per_segment;
    const int group_keys_begin_idx =
      (worker_thread_idx / config.threads_per_segment) * config.threads_per_segment * ItemsPerThread;
    const bool active               = segment_idx != ~0u;
    const unsigned input_segment_id = active ? bins.segment_ids[segment_idx] : 0;
    const auto begin =
      active ? begins[input_segment_id] : ::cuda::std::remove_cvref_t<decltype(begins[input_segment_id])>{};
    const int segment_size = active ? static_cast<int>(ends[input_segment_id] - begin) : 0;
    Key keys[ItemsPerThread];
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int local                           = item * config.threads_per_segment + thread_idx_in_group;
      keys_shared[group_keys_begin_idx + local] = local < segment_size ? in[begin + local] : Key{};
    }
    // SerialMerge prefetches one unused key beyond its runs.
    if (worker_thread_idx == 0)
    {
      keys_shared[threads_per_worker * ItemsPerThread] = Key{};
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys[item] = keys_shared[group_keys_begin_idx + thread_idx_in_group * ItemsPerThread + item];
    }
    if (thread_idx_in_group * ItemsPerThread < segment_size)
    {
      Key max_key = keys[0];
      for (int item = 1; item < ItemsPerThread; ++item)
      {
        if (thread_idx_in_group * ItemsPerThread + item < segment_size)
        {
          max_key = compare(max_key, keys[item]) ? keys[item] : max_key;
        }
        else
        {
          keys[item] = max_key;
        }
      }
      NullType values[ItemsPerThread];
      StableOddEvenSort(keys, values, compare);
    }
    for (int merged_threads = 2; merged_threads <= config.threads_per_segment; merged_threads *= 2)
    {
      sync();
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        keys_shared[worker_thread_idx * ItemsPerThread + item] = keys[item];
      }
      sync();
      const int start     = (worker_thread_idx & ~(merged_threads - 1)) * ItemsPerThread;
      const int valid_end = group_keys_begin_idx + segment_size;
      const int first     = (::cuda::std::min) (valid_end, start);
      const int middle    = (::cuda::std::min) (valid_end, first + merged_threads / 2 * ItemsPerThread);
      const int last      = (::cuda::std::min) (valid_end, middle + merged_threads / 2 * ItemsPerThread);
      const int diag = (::cuda::std::min) (last - first, (worker_thread_idx & (merged_threads - 1)) * ItemsPerThread);
      const int split =
        MergePath(keys_shared + first, keys_shared + middle, middle - first, last - middle, diag, compare);
      int indices[ItemsPerThread];
      SerialMerge(
        keys_shared,
        first + split,
        middle + diag - split,
        middle - first - split,
        last - middle - diag + split,
        keys,
        indices,
        compare,
        Key{});
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys_shared[worker_thread_idx * ItemsPerThread + item] = keys[item];
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int local = item * config.threads_per_segment + thread_idx_in_group;
      if (local < segment_size)
      {
        out[begin + local] = keys_shared[group_keys_begin_idx + local];
      }
    }
    sync();
  }

  // Translate runtime IPT into a specialization with a fixed-size per-thread key array.
  template <typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void invoke(
    unsigned segment_idx,
    bin_config config,
    const partition_result& bins,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    Key* keys_shared)
  {
    // Select the fixed-size per-thread arrays for the chosen IPT.
    switch (config.items_per_thread)
    {
      case 9:
        sort_segment<9>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 10:
        sort_segment<10>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 11:
        sort_segment<11>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 12:
        sort_segment<12>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 13:
        sort_segment<13>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 14:
        sort_segment<14>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 15:
        sort_segment<15>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      default:
        _CCCL_UNREACHABLE();
    }
  }
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END

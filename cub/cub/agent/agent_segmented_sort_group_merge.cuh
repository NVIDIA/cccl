// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
#include <cub/detail/segmented_sort_capacity.cuh>
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
  _CCCL_HOST_DEVICE_API static constexpr bin_config expanded_config(int bin)
  {
    // Enumerate supported pairs at compile time, without encoding bin positions.
    for (int threads = policy.min_threads; threads <= policy.max_threads; threads *= 2)
    {
      for (int ipt = policy.min_ipt; ipt <= policy.max_ipt; ++ipt)
      {
        if (bin-- == 0)
        {
          return {threads, ipt};
        }
      }
    }
    _CCCL_UNREACHABLE();
  }
  // Build the capacity-list type from threads * IPT for every supported bin.
  template <::cuda::std::size_t... BinIndices>
  static auto expand(::cuda::std::index_sequence<BinIndices...>)
    -> capacity_list<(expanded_config(BinIndices).threads * expanded_config(BinIndices).items_per_thread)...>;
  using capacities = decltype(expand(::cuda::std::make_index_sequence<size>{}));
  // Expose a selected pair as the configuration type expected by the generic bin lookup.
  template <int Bin>
  struct bin_selection
  {
    inline static constexpr bin_config chosen = expanded_config(Bin);
    using type                                = bin_shape<chosen.threads, chosen.items_per_thread>;
  };
  template <int Bin>
  using config_for_bin = typename bin_selection<Bin>::type;
};

// Adapted from call_group_merge_runtime_impl (sort_block_combined.cuh) and
// merge_warp_groups_runtime (merge_warp_groups.cuh) in the LRB prototype.
// This ascending numeric worker omits the FP-twiddle and half-presort paths.
// Warp work units have independent scratch and never use CTA barriers.
template <typename PolicyGetter>
struct group_merge_implementation
{
  inline static constexpr auto policy = PolicyGetter{}();
  using expansion                     = group_merge_expansion<PolicyGetter>;
  static constexpr bool is_warp       = policy.max_threads <= 32;
  static constexpr int unit_threads   = is_warp ? 32 : policy.block_threads;
  static_assert(policy.block_threads % 32 == 0);
  static_assert(is_warp || policy.min_threads >= 64);
  static_assert(policy.block_threads % policy.max_threads == 0);
  static_assert(policy.min_ipt >= 5 && policy.max_ipt <= 15, "GroupMerge workers support IPT 5..15");

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

  // Sort each cooperating group's segment: load, sort thread-local runs, merge, and store.
  // A slot of ~0u masks an unused group while keeping its threads in synchronization.
  // NaN ordering is unspecified; -0 and +0 compare equal and their relative order
  // is not guaranteed. This does not match radix-sort ordering used for large
  // segments in DeviceSegmentedSort. Decide the contract before public dispatch.
  template <int ItemsPerThread, typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void sort_segment(
    unsigned slot,
    bin_config config,
    const partition_result& bins,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    Key* scratch)
  {
    const ::cuda::std::less<Key> compare{};
    const int lane = threadIdx.x % unit_threads;
    if constexpr (is_warp)
    {
      scratch += (threadIdx.x / 32) * (32 * policy.max_ipt + 1);
    }
    const int group_lane   = lane % config.threads;
    const int group_base   = (lane / config.threads) * config.threads * ItemsPerThread;
    const bool active      = slot != ~0u;
    const unsigned segment = active ? bins.segment_ids[slot] : 0;
    const auto begin       = active ? begins[segment] : ::cuda::std::remove_cvref_t<decltype(begins[segment])>{};
    const int count        = active ? static_cast<int>(ends[segment] - begin) : 0;
    Key keys[ItemsPerThread];
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int local             = item * config.threads + group_lane;
      scratch[group_base + local] = local < count ? in[begin + local] : Key{};
    }
    // SerialMerge prefetches one unused key beyond its runs.
    if (lane == 0)
    {
      scratch[unit_threads * ItemsPerThread] = Key{};
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys[item] = scratch[group_base + group_lane * ItemsPerThread + item];
    }
    if (group_lane * ItemsPerThread < count)
    {
      Key max_key = keys[0];
      for (int item = 1; item < ItemsPerThread; ++item)
      {
        if (group_lane * ItemsPerThread + item < count)
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
    for (int merged_threads = 2; merged_threads <= config.threads; merged_threads *= 2)
    {
      sync();
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        scratch[lane * ItemsPerThread + item] = keys[item];
      }
      sync();
      const int start     = (lane & ~(merged_threads - 1)) * ItemsPerThread;
      const int valid_end = group_base + count;
      const int first     = (::cuda::std::min) (valid_end, start);
      const int middle    = (::cuda::std::min) (valid_end, first + merged_threads / 2 * ItemsPerThread);
      const int last      = (::cuda::std::min) (valid_end, middle + merged_threads / 2 * ItemsPerThread);
      const int diag      = (::cuda::std::min) (last - first, (lane & (merged_threads - 1)) * ItemsPerThread);
      const int split     = MergePath(scratch + first, scratch + middle, middle - first, last - middle, diag, compare);
      int indices[ItemsPerThread];
      SerialMerge(
        scratch,
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
      scratch[lane * ItemsPerThread + item] = keys[item];
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int local = item * config.threads + group_lane;
      if (local < count)
      {
        out[begin + local] = scratch[group_base + local];
      }
    }
    sync();
  }

  // Translate runtime IPT into a specialization with a fixed-size per-thread key array.
  template <typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void invoke(
    unsigned slot,
    bin_config config,
    const partition_result& bins,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    Key* scratch)
  {
    // Match the source workers: select a fixed register-array size with a switch.
    switch (config.items_per_thread)
    {
      case 5:
        sort_segment<5>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 6:
        sort_segment<6>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 7:
        sort_segment<7>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 8:
        sort_segment<8>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 9:
        sort_segment<9>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 10:
        sort_segment<10>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 11:
        sort_segment<11>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 12:
        sort_segment<12>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 13:
        sort_segment<13>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 14:
        sort_segment<14>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      case 15:
        sort_segment<15>(slot, config, bins, begins, ends, in, out, scratch);
        break;
      default:
        _CCCL_UNREACHABLE();
    }
  }
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END

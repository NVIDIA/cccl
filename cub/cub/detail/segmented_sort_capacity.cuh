// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Contract between partitioning and sorting: capacity bounds, bin configurations, and partition results.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_namespace.cuh>

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Ordered capacity bounds supplied by sorting entries; partition finds the smallest fitting bin.
template <int... Bounds>
struct capacity_list
{
  static constexpr int size = sizeof...(Bounds);
  inline static constexpr ::cuda::std::array<int, size> bounds{{Bounds...}};

  // Require nonempty, positive, strictly increasing bounds for binary-search routing.
  _CCCL_HOST_DEVICE_API static constexpr bool valid()
  {
    for (int i = 0; i < size; ++i)
    {
      if (bounds[i] <= 0 || (i && bounds[i - 1] >= bounds[i]))
      {
        return false;
      }
    }
    return size > 0;
  }
  static_assert(valid(), "capacities must be positive and strictly increasing");

  // Find the smallest capacity covering length; size denotes the implicit overflow bin.
  template <int Begin = 0, int End = size, typename Offset>
  _CCCL_HOST_DEVICE_API static constexpr int first_fit([[maybe_unused]] Offset length)
  {
    if constexpr (Begin == End)
    {
      return Begin;
    }
    else
    {
      constexpr int mid = Begin + (End - Begin) / 2;
      return length <= bounds[mid] ? first_fit<Begin, mid>(length) : first_fit<mid + 1, End>(length);
    }
  }
};

// Flatten entry capacities in launch order, preserving each entry's bin order.
template <typename... Lists>
struct concatenate_capacities;
template <typename List>
struct concatenate_capacities<List>
{
  using type = List;
};
template <int... Left, int... Right, typename... Rest>
struct concatenate_capacities<capacity_list<Left...>, capacity_list<Right...>, Rest...>
    : concatenate_capacities<capacity_list<Left..., Right...>, Rest...>
{};

// Encode one bin's configuration in a type for compile-time mapping.
// group_merge_expansion::config_for_bin<Bin> supplies this type to the lookup below.
template <int Threads, int ItemsPerThread>
struct bin_shape
{
  static constexpr int threads          = Threads;
  static constexpr int items_per_thread = ItemsPerThread;
};

// Configuration values returned by config_for_bin to sort_entry_kernel for work packing
// and passed to the GroupMerge worker. Also used during compile-time capacity expansion.
struct bin_config
{
  int threads;
  int items_per_thread;
};

// An expansion enumerates a sorter's bins: capacities and a config_for_bin<Bin> type.
// The runtime local bin selects a leaf whose configuration is known at compile time.
template <typename Expansion, int Begin = 0, int End = Expansion::capacities::size>
_CCCL_HOST_DEVICE_API bin_config config_for_bin(int bin)
{
  static_assert(Begin < End);
  if constexpr (End - Begin == 1)
  {
    using config = typename Expansion::template config_for_bin<Begin>;
    return {config::threads, config::items_per_thread};
  }
  else
  {
    constexpr int mid = Begin + (End - Begin) / 2;
    return bin < mid ? config_for_bin<Expansion, Begin, mid>(bin) : config_for_bin<Expansion, mid, End>(bin);
  }
}

// Describe one sorting launch: its worker, supported bins, and launch settings.
template <typename Implementation>
struct sort_entry
{
  using implementation                = Implementation;
  using expansion                     = typename implementation::expansion;
  using capacities                    = typename expansion::capacities;
  inline static constexpr auto tuning = implementation::policy;
  // Threads per independently scheduled work unit (warp or block), which may pack several segments.
  static constexpr int unit_threads = implementation::unit_threads;
  static_assert(tuning.block_threads > 0 && tuning.block_threads <= 1024);
  static_assert(unit_threads > 0 && tuning.block_threads % unit_threads == 0);
  static_assert(tuning.blocks_per_sm > 0 && tuning.min_blocks_per_sm > 0);
  static_assert(capacities::valid());
  // Let the kernel obtain a local bin's configuration through its entry.
  _CCCL_HOST_DEVICE_API static bin_config config(int bin)
  {
    return config_for_bin<expansion>(bin);
  }
};

// Combine launches into one partition capacity list; each entry owns a consecutive range.
template <typename... Entries>
struct sort_entries
{
  static constexpr int size = sizeof...(Entries);
  static_assert(size > 0);
  using entries = ::cuda::std::tuple<Entries...>;
  template <::cuda::std::size_t EntryIndex>
  using entry      = ::cuda::std::tuple_element_t<EntryIndex, entries>;
  using capacities = typename concatenate_capacities<typename Entries::capacities...>::type;
  static_assert(capacities::valid(),
                "Capacities of consecutive sort entries must be strictly increasing: the largest capacity of entry N "
                "(max_threads * max_ipt) must be below the smallest of entry N+1 (min_threads * min_ipt).");
  // First bin of this entry in the flattened list; entry-local bin indices start at zero.
  _CCCL_HOST_DEVICE_API static constexpr int bin_begin(int entry_index)
  {
    constexpr int counts[] = {Entries::capacities::size...};
    int result             = 0;
    for (int i = 0; i < entry_index; ++i)
    {
      result += counts[i];
    }
    return result;
  }
};

// Non-owning views into temporary storage; all bins share one segment-ID queue.
struct partition_result
{
  unsigned* bin_offsets{}; // Bounded bin count + 2 offsets, including overflow's end.
  unsigned* segment_ids{};
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END

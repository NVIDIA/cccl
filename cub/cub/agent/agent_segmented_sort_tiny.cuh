// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sort segments shorter than the tiny-walk cutoff with one logical warp per segment.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/functional>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Out-of-place keys-only sort of one segment per logical warp of 1, 2, or 4 threads.
template <typename Key>
struct tiny_sort_implementation
{
  using policy                          = tiny_walk_policy;
  static constexpr int items_per_thread = policy::items_per_thread;
  static_assert(policy::max_threads == 4, "temp_storage provides logical warps of 1, 2, and 4 threads");
  static_assert(policy::block_threads % policy::max_threads == 0);

  template <int LogicalWarpThreads>
  struct logical_warp
  {
    using load_t  = WarpLoad<Key, items_per_thread, WARP_LOAD_TRANSPOSE, LogicalWarpThreads>;
    using sort_t  = WarpMergeSort<Key, items_per_thread, LogicalWarpThreads>;
    using store_t = WarpStore<Key, items_per_thread, WARP_STORE_DIRECT, LogicalWarpThreads>;
    union temp_storage
    {
      typename load_t::TempStorage load;
      typename sort_t::TempStorage sort;
      typename store_t::TempStorage store;
    };
  };

  // Every logical warp size reuses the same bytes, with one slot per logical warp in the block.
  union temp_storage
  {
    typename logical_warp<1>::temp_storage threads_1[policy::block_threads];
    typename logical_warp<2>::temp_storage threads_2[policy::block_threads / 2];
    typename logical_warp<4>::temp_storage threads_4[policy::block_threads / 4];
  };

  template <int LogicalWarpThreads>
  _CCCL_DEVICE_API static typename logical_warp<LogicalWarpThreads>::temp_storage& slot(temp_storage& storage)
  {
    const int logical_warp_idx = threadIdx.x / LogicalWarpThreads;
    if constexpr (LogicalWarpThreads == 1)
    {
      return storage.threads_1[logical_warp_idx];
    }
    else if constexpr (LogicalWarpThreads == 2)
    {
      return storage.threads_2[logical_warp_idx];
    }
    else
    {
      static_assert(LogicalWarpThreads == 4);
      return storage.threads_4[logical_warp_idx];
    }
  }

  // Every thread of the warp calls this with the same LogicalWarpThreads. Logical warps without a
  // segment pass is_valid = false and still take part in the collectives with zero keys.
  // The segment holds at most LogicalWarpThreads * items_per_thread keys. Callers synchronize the
  // warp before storage is reused.
  template <int LogicalWarpThreads, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void sort_segment(
    temp_storage& storage,
    unsigned segment_id,
    bool is_valid,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out)
  {
    using warp_t       = logical_warp<LogicalWarpThreads>;
    auto& slot_tile    = slot<LogicalWarpThreads>(storage);
    const auto begin   = is_valid ? begins[segment_id] : ::cuda::std::remove_cvref_t<decltype(begins[segment_id])>{};
    const int num_keys = is_valid ? static_cast<int>(ends[segment_id] - begin) : 0;
    typename warp_t::sort_t sort(slot_tile.sort);
    Key keys[items_per_thread];
    typename warp_t::load_t(slot_tile.load).Load(try_make_cache_modified_iterator<LOAD_LDG>(in + begin), keys, num_keys);
    // Every lane calls this, so sync the full warp: a per-logical-warp mask differs across lanes and
    // compiles to a MATCH.ANY / REDUX sequence.
    __syncwarp();
    sort.Sort(keys, ::cuda::std::less<Key>{}, num_keys);
    __syncwarp();
    typename warp_t::store_t(slot_tile.store).Store(out + begin, keys, num_keys);
  }
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END

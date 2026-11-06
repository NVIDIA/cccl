// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device kernels for the tiny-segment walk, capacity-based ID partitioning, and sorting with work prefixes
// built in shared memory.

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_sort_lrb.cuh>
#include <cub/agent/agent_segmented_sort_tiny.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/detail/segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{
// Count nonempty segments by capacity, or scatter their IDs into reserved per-bin ranges.
// Partition receives capacity bounds only; it never receives sorting configurations.
template <typename Capacities,
          bool Scatter,
          typename BeginIterator,
          typename EndIterator,
          typename PolicySelector = partition_policy_selector>
__launch_bounds__(current_policy<PolicySelector>().threads_per_block) _CCCL_KERNEL_ATTRIBUTES void partition_kernel(
  BeginIterator begins,
  EndIterator ends,
  unsigned num_segments,
  unsigned* counts_or_cursors,
  ::cuda::std::conditional_t<Scatter, unsigned, const unsigned>* ids)
{
  constexpr int total_bins = Capacities::size + 1;
  __shared__ unsigned bin_counts_shared[total_bins];
  [[maybe_unused]] __shared__ unsigned bin_starts_shared[total_bins];
  for (int bin_idx = static_cast<int>(threadIdx.x); bin_idx < total_bins; bin_idx += static_cast<int>(blockDim.x))
  {
    bin_counts_shared[bin_idx] = 0;
  }
  __syncthreads();
  const unsigned seg_id                         = (blockIdx.x * blockDim.x) + threadIdx.x;
  int bin_idx                                   = -1;
  [[maybe_unused]] unsigned segment_rank_in_bin = 0;
  if (seg_id < num_segments)
  {
    const auto begin = begins[seg_id];
    const auto end   = ends[seg_id];
    if (end > begin)
    {
      bin_idx             = Capacities::first_fit(end - begin);
      segment_rank_in_bin = atomicAdd(&bin_counts_shared[bin_idx], 1u);
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < total_bins; i += static_cast<int>(blockDim.x))
  {
    if (bin_counts_shared[i])
    {
      bin_starts_shared[i] = atomicAdd(&counts_or_cursors[i], bin_counts_shared[i]);
    }
  }
  if constexpr (Scatter)
  {
    __syncthreads();
    if (bin_idx >= 0)
    {
      ids[bin_starts_shared[bin_idx] + segment_rank_in_bin] = seg_id;
    }
  }
}

// Each block walks one tile of threads_per_block * segments_per_thread segments. Empty segments are skipped,
// singletons are copied, and segments shorter than inline_cutoff are queued by logical warp size and sorted
// by the block. Longer segments are counted by capacity; the block flushes its counts only if it found one.
template <typename Capacities, typename BeginIterator, typename EndIterator, typename Key>
__launch_bounds__(tiny_walk_policy::threads_per_block) _CCCL_KERNEL_ATTRIBUTES void tiny_walk_kernel(
  BeginIterator begins, EndIterator ends, unsigned num_segments, const Key* in, Key* out, unsigned* counts)
{
  using policy_t                   = tiny_walk_policy;
  using sorter_t                   = tiny_sort_implementation<Key>;
  constexpr int total_bins         = Capacities::size + 1;
  constexpr unsigned tile_segments = policy_t::threads_per_block * policy_t::segments_per_thread;
  // One queue each for logical warps of 1, 2, and 4 threads.
  constexpr int num_queues = 3;
  __shared__ unsigned bin_counts_shared[total_bins];
  __shared__ unsigned queue_sizes_shared[num_queues];
  __shared__ unsigned queues_shared[num_queues][tile_segments];
  __shared__ typename sorter_t::temp_storage sort_storage;
  for (int bin_idx = static_cast<int>(threadIdx.x); bin_idx < total_bins; bin_idx += static_cast<int>(blockDim.x))
  {
    bin_counts_shared[bin_idx] = 0;
  }
  if (threadIdx.x < num_queues)
  {
    queue_sizes_shared[threadIdx.x] = 0;
  }
  __syncthreads();
  const unsigned tile_begin = blockIdx.x * tile_segments;
  // Compare tile-local indices so that the last tile cannot overflow unsigned segment IDs.
  const unsigned tile_size = (::cuda::std::min) (tile_segments, num_segments - tile_begin);
  bool has_deferred        = false;
  for (int pass = 0; pass < policy_t::segments_per_thread; ++pass)
  {
    const unsigned idx_in_tile = (pass * policy_t::threads_per_block) + threadIdx.x;
    if (idx_in_tile < tile_size)
    {
      const unsigned segment_id = tile_begin + idx_in_tile;
      const auto begin          = begins[segment_id];
      const auto length         = ends[segment_id] - begin;
      if (length == 1)
      {
        out[begin] = in[begin];
      }
      else if (length > 1 && length < policy_t::inline_cutoff)
      {
        const int queue = length <= policy_t::items_per_thread ? 0 : length <= 2 * policy_t::items_per_thread ? 1 : 2;
        queues_shared[queue][atomicAdd(&queue_sizes_shared[queue], 1u)] = segment_id;
      }
      else if (length >= policy_t::inline_cutoff)
      {
        atomicAdd(&bin_counts_shared[Capacities::first_fit(length)], 1u);
        has_deferred = true;
      }
    }
  }
  if (__syncthreads_or(has_deferred))
  {
    for (int bin_idx = static_cast<int>(threadIdx.x); bin_idx < total_bins; bin_idx += static_cast<int>(blockDim.x))
    {
      if (bin_counts_shared[bin_idx])
      {
        atomicAdd(&counts[bin_idx], bin_counts_shared[bin_idx]);
      }
    }
  }
  // Logical warp sizes alias sort_storage, so every pass ends with a block barrier.
  const auto sort_queue = [&](auto logical_warp_threads, int queue) {
    constexpr int threads                = decltype(logical_warp_threads)::value;
    constexpr unsigned segments_per_pass = policy_t::threads_per_block / threads;
    const unsigned queue_size            = queue_sizes_shared[queue];
    for (unsigned pass_begin = 0; pass_begin < queue_size; pass_begin += segments_per_pass)
    {
      const unsigned queue_idx = pass_begin + (threadIdx.x / threads);
      const bool is_valid      = queue_idx < queue_size;
      sorter_t::template sort_segment<threads>(
        sort_storage, is_valid ? queues_shared[queue][queue_idx] : 0, is_valid, begins, ends, in, out);
      __syncthreads();
    }
  };
  sort_queue(::cuda::std::integral_constant<int, 1>{}, 0);
  sort_queue(::cuda::std::integral_constant<int, 2>{}, 1);
  sort_queue(::cuda::std::integral_constant<int, 4>{}, 2);
}

// One warp converts bin counts into global ID offsets and initial scatter cursors.
template <int NumBins>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void scan_bin_counts(const unsigned* counts, unsigned* offsets, unsigned* cursors)
{
  constexpr int items_per_thread = ::cuda::ceil_div(NumBins, detail::warp_threads);
  using load_t                   = WarpLoad<unsigned, items_per_thread, WARP_LOAD_TRANSPOSE>;
  using scan_t                   = BlockScan<unsigned, detail::warp_threads>;
  using store_t                  = WarpStore<unsigned, items_per_thread, WARP_STORE_TRANSPOSE>;
  __shared__ typename load_t::TempStorage load_storage;
  __shared__ scan_t::TempStorage scan_storage;
  __shared__ typename store_t::TempStorage store_storage;
  unsigned items[items_per_thread];
  unsigned prefixes[items_per_thread];
  unsigned cursor_prefixes[items_per_thread];
  unsigned total;
  load_t(load_storage).Load(counts, items, NumBins, 0u);
  scan_t(scan_storage).ExclusiveSum(items, prefixes, total);
  // WarpStore transposes its input registers; retain a copy for the second output.
  for (int item = 0; item < items_per_thread; ++item)
  {
    cursor_prefixes[item] = prefixes[item];
  }
  store_t(store_storage).Store(offsets, prefixes, NumBins);
  __syncwarp(); // Reuse the transpose storage after every lane has read it.
  store_t(store_storage).Store(cursors, cursor_prefixes, NumBins);
  if (threadIdx.x == 0)
  {
    offsets[NumBins] = total;
  }
}

template <typename Capacities>
_CCCL_KERNEL_ATTRIBUTES void segment_prefix_kernel(const unsigned* counts, unsigned* offsets, unsigned* cursors)
{
  scan_bin_counts<Capacities::size + 1>(counts, offsets, cursors);
}

// Consume this kernel's bins, packing segments into groups and invoking its sorting implementation.
// The kernel type carries the selector; launch bounds and the body select tuning for the compiled target.
template <typename Kernel, typename Key, typename BeginIterator, typename EndIterator>
__launch_bounds__(current_policy<typename Kernel::policy_selector>().threads_per_block,
                  current_policy<typename Kernel::policy_selector>().min_resident_blocks_per_sm)
  _CCCL_KERNEL_ATTRIBUTES void sort_kernel(
    partition_result partition, BeginIterator begins, EndIterator ends, const Key* in, Key* out)
{
  using implementation_t = typename Kernel::template implementation_t<
    device_policy_getter<typename Kernel::policy_selector, current_tuning_cc().get()>>;
  using agent = agent_t<implementation_t>;
  __shared__ typename agent::TempStorage temp_storage;
  extern __shared__ __align__(256) unsigned char dynamic_smem[];
  agent::process(partition, begins, ends, in, out, temp_storage, reinterpret_cast<Key*>(dynamic_smem));
}
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END

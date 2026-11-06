// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device kernels for the tiny-segment walk, capacity-based ID partitioning, and sorting with work prefixes
// built in shared memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_sort_tiny.cuh>
#include <cub/detail/segmented_sort_entries.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_arch.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Count segments of at least MinLength keys by capacity, or scatter their IDs into reserved per-bin ranges.
// Count and scatter must use the same MinLength. Partition receives capacity bounds only; it never receives
// sorting configurations.
template <typename Capacities, bool Scatter, typename BeginIterator, typename EndIterator, int MinLength = 1>
__launch_bounds__(partition_policy::block_threads) _CCCL_KERNEL_ATTRIBUTES void partition_kernel(
  BeginIterator begins, EndIterator ends, unsigned num_segments, unsigned* counts_or_cursors, unsigned* ids)
{
  constexpr unsigned total_bins = Capacities::size + 1;
  __shared__ unsigned bin_counts_shared[total_bins];
  [[maybe_unused]] __shared__ unsigned bin_starts_shared[total_bins];
  for (unsigned bin_idx = threadIdx.x; bin_idx < total_bins; bin_idx += blockDim.x)
  {
    bin_counts_shared[bin_idx] = 0;
  }
  __syncthreads();
  const unsigned input_segment_id               = blockIdx.x * blockDim.x + threadIdx.x;
  int bin_idx                                   = -1;
  [[maybe_unused]] unsigned segment_rank_in_bin = 0;
  if (input_segment_id < num_segments)
  {
    const auto begin = begins[input_segment_id];
    const auto end   = ends[input_segment_id];
    // Compare before subtracting: with unsigned offsets, end < begin would wrap to a huge length.
    if (end > begin && end - begin >= MinLength)
    {
      bin_idx             = Capacities::first_fit(end - begin);
      segment_rank_in_bin = atomicAdd(bin_counts_shared + bin_idx, 1u);
    }
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < total_bins; i += blockDim.x)
  {
    if (bin_counts_shared[i])
    {
      bin_starts_shared[i] = atomicAdd(counts_or_cursors + i, bin_counts_shared[i]);
    }
  }
  if constexpr (Scatter)
  {
    __syncthreads();
    if (bin_idx >= 0)
    {
      ids[bin_starts_shared[bin_idx] + segment_rank_in_bin] = input_segment_id;
    }
  }
}

// Each block walks one tile of block_threads * segments_per_thread segments. Empty segments are skipped,
// singletons are copied, and segments shorter than inline_cutoff are queued by logical warp size and sorted
// by the block. Longer segments are counted by capacity; the block flushes its counts only if it found one.
template <typename Capacities, typename BeginIterator, typename EndIterator, typename Key>
__launch_bounds__(tiny_walk_policy::block_threads) _CCCL_KERNEL_ATTRIBUTES void tiny_walk_kernel(
  BeginIterator begins, EndIterator ends, unsigned num_segments, const Key* in, Key* out, unsigned* counts)
{
  using policy_t                   = tiny_walk_policy;
  using sorter_t                   = tiny_sort_implementation<Key>;
  constexpr unsigned total_bins    = Capacities::size + 1;
  constexpr unsigned tile_segments = policy_t::block_threads * policy_t::segments_per_thread;
  // One queue each for logical warps of 1, 2, and 4 threads.
  constexpr int num_queues = 3;
  __shared__ unsigned bin_counts_shared[total_bins];
  __shared__ unsigned queue_sizes_shared[num_queues];
  __shared__ unsigned queues_shared[num_queues][tile_segments];
  __shared__ typename sorter_t::temp_storage sort_storage;
  for (unsigned bin_idx = threadIdx.x; bin_idx < total_bins; bin_idx += blockDim.x)
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
    const unsigned idx_in_tile = pass * policy_t::block_threads + threadIdx.x;
    if (idx_in_tile < tile_size)
    {
      const unsigned segment_id = tile_begin + idx_in_tile;
      const auto begin          = begins[segment_id];
      const auto end            = ends[segment_id];
      // Compare before subtracting: with unsigned offsets, end < begin would wrap to a huge length.
      const auto length = end > begin ? end - begin : decltype(end - begin){};
      if (length == 1)
      {
        out[begin] = in[begin];
      }
      else if (length > 1 && length < policy_t::inline_cutoff)
      {
        const int queue = length <= policy_t::items_per_thread ? 0 : length <= 2 * policy_t::items_per_thread ? 1 : 2;
        queues_shared[queue][atomicAdd(queue_sizes_shared + queue, 1u)] = segment_id;
      }
      else if (length >= policy_t::inline_cutoff)
      {
        atomicAdd(bin_counts_shared + Capacities::first_fit(length), 1u);
        has_deferred = true;
      }
    }
  }
  if (__syncthreads_or(has_deferred))
  {
    for (unsigned bin_idx = threadIdx.x; bin_idx < total_bins; bin_idx += blockDim.x)
    {
      if (bin_counts_shared[bin_idx])
      {
        atomicAdd(counts + bin_idx, bin_counts_shared[bin_idx]);
      }
    }
  }
  // Logical warp sizes alias sort_storage, so every pass ends with a block barrier.
  const auto sort_queue = [&](auto logical_warp_threads, int queue) {
    constexpr int threads                = decltype(logical_warp_threads)::value;
    constexpr unsigned segments_per_pass = policy_t::block_threads / threads;
    const unsigned queue_size            = queue_sizes_shared[queue];
    for (unsigned pass_begin = 0; pass_begin < queue_size; pass_begin += segments_per_pass)
    {
      const unsigned queue_idx = pass_begin + threadIdx.x / threads;
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

// One thread converts bin counts into global ID offsets and initial scatter cursors.
template <typename Capacities>
_CCCL_KERNEL_ATTRIBUTES void segment_prefix_kernel(const unsigned* counts, unsigned* offsets, unsigned* cursors)
{
  unsigned prefix = 0;
  for (int bin_idx = 0; bin_idx <= Capacities::size; ++bin_idx)
  {
    offsets[bin_idx] = cursors[bin_idx] = prefix;
    prefix += counts[bin_idx];
  }
  offsets[Capacities::size + 1] = prefix;
}

// Consume this kernel's bins, packing segments into groups and invoking its sorting implementation.
// The kernel type carries the selector; launch bounds and the body select tuning for the compiled target.
template <typename Kernel, typename Key, typename BeginIterator, typename EndIterator>
__launch_bounds__(current_policy<typename Kernel::policy_selector>().block_threads,
                  current_policy<typename Kernel::policy_selector>().min_blocks_per_sm)
  _CCCL_KERNEL_ATTRIBUTES void sort_kernel(
    partition_result partition, BeginIterator begins, EndIterator ends, const Key* in, Key* out)
{
  using implementation_t = typename Kernel::template implementation_t<
    device_policy_getter<typename Kernel::policy_selector, current_tuning_cc().get()>>;
  using capacities_t                   = typename implementation_t::expansion::capacities;
  constexpr auto tuning                = current_policy<typename Kernel::policy_selector>();
  constexpr int num_kernel_bins        = capacities_t::size;
  constexpr unsigned threads_per_group = implementation_t::threads_per_group;
  constexpr unsigned groups_per_block  = tuning.block_threads / threads_per_group;
  if (static_cast<unsigned long long>(blockIdx.x) * groups_per_block
      >= partition.bin_offsets[num_kernel_bins] - partition.bin_offsets[0])
  {
    return;
  }
  // Convert per-bin segment counts into work-item counts; each block builds its own prefix.
  __shared__ unsigned bin_work_offsets[num_kernel_bins + 1];
  if (threadIdx.x == 0)
  {
    unsigned prefix     = 0;
    bin_work_offsets[0] = 0;
    for (int bin_idx = 0; bin_idx < num_kernel_bins; ++bin_idx)
    {
      const auto config                  = lookup_bin_config<typename implementation_t::expansion>(bin_idx);
      const unsigned segments_per_group  = threads_per_group / config.threads_per_segment;
      const unsigned num_segments_in_bin = partition.bin_offsets[bin_idx + 1] - partition.bin_offsets[bin_idx];
      prefix += num_segments_in_bin / segments_per_group + (num_segments_in_bin % segments_per_group != 0);
      bin_work_offsets[bin_idx + 1] = prefix;
    }
  }
  __syncthreads();
  // Key storage used by the sorting groups, separate from the work prefix.
  extern __shared__ __align__(256) unsigned char dynamic_smem[];
  const unsigned thread_idx_in_group = threadIdx.x % threads_per_group;
  const auto first_work_id =
    static_cast<unsigned long long>(blockIdx.x) * groups_per_block + threadIdx.x / threads_per_group;
  const auto work_stride = static_cast<unsigned long long>(gridDim.x) * groups_per_block;
  // Each work item assigns one group up to segments_per_group segments from one bin.
  for (auto work_id = first_work_id; work_id < bin_work_offsets[num_kernel_bins]; work_id += work_stride)
  {
    // Locate this work ID's bin in the shared-memory prefix.
    int first = 1;
    int last  = num_kernel_bins;
    while (first < last)
    {
      const int mid = first + (last - first) / 2;
      if (work_id >= bin_work_offsets[mid])
      {
        first = mid + 1;
      }
      else
      {
        last = mid;
      }
    }
    const int bin_idx                 = first - 1;
    const auto config                 = lookup_bin_config<typename implementation_t::expansion>(bin_idx);
    const unsigned segments_per_group = threads_per_group / config.threads_per_segment;
    const auto segment_idx = partition.bin_offsets[bin_idx] + (work_id - bin_work_offsets[bin_idx]) * segments_per_group
                           + thread_idx_in_group / config.threads_per_segment;
    implementation_t::invoke(
      segment_idx < partition.bin_offsets[bin_idx + 1] ? segment_idx : ~0u,
      config,
      partition,
      begins,
      ends,
      in,
      out,
      reinterpret_cast<Key*>(dynamic_smem));
  }
}
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END

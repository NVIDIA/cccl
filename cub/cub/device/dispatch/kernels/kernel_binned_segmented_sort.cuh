// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device kernels for capacity-based ID partitioning and sorting with work prefixes built in shared memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/segmented_sort_entries.cuh>
#include <cub/util_arch.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Count nonempty segments by capacity, or scatter their IDs into reserved per-bin ranges.
// The selector supplies capacity bounds only; partition never receives sorting configurations.
template <typename CapacitySelector, bool Scatter, typename BeginIterator, typename EndIterator>
__launch_bounds__(256) _CCCL_KERNEL_ATTRIBUTES void partition_kernel(
  BeginIterator begins, EndIterator ends, unsigned num_segments, unsigned* counts_or_cursors, unsigned* ids)
{
  using capacities_t            = typename CapacitySelector::template capacities_t<device_compute_capability_getter_t>;
  constexpr unsigned total_bins = capacities_t::size + 1;
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
    if (end > begin)
    {
      bin_idx             = capacities_t::first_fit(end - begin);
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

// One thread converts bin counts into global ID offsets and initial scatter cursors.
template <typename CapacitySelector>
_CCCL_KERNEL_ATTRIBUTES void segment_prefix_kernel(const unsigned* counts, unsigned* offsets, unsigned* cursors)
{
  using capacities_t = typename CapacitySelector::template capacities_t<device_compute_capability_getter_t>;
  unsigned prefix    = 0;
  for (int bin_idx = 0; bin_idx <= capacities_t::size; ++bin_idx)
  {
    offsets[bin_idx] = cursors[bin_idx] = prefix;
    prefix += counts[bin_idx];
  }
  offsets[capacities_t::size + 1] = prefix;
}

// Consume this kernel's bins, packing segments into workers and invoking its sorting implementation.
// The kernel type carries the selector; launch bounds and the body select tuning for the compiled target.
template <typename Kernel, typename Key, typename BeginIterator, typename EndIterator>
__launch_bounds__(current_policy<typename Kernel::policy_selector>().block_threads,
                  current_policy<typename Kernel::policy_selector>().min_blocks_per_sm)
  _CCCL_KERNEL_ATTRIBUTES void sort_kernel(
    partition_result partition, BeginIterator begins, EndIterator ends, const Key* in, Key* out)
{
  using implementation_t        = typename Kernel::template implementation_t<device_compute_capability_getter_t>;
  using capacities_t            = typename implementation_t::expansion::capacities;
  constexpr auto tuning         = current_policy<typename Kernel::policy_selector>();
  constexpr int num_kernel_bins = capacities_t::size;
  constexpr unsigned threads_per_worker = implementation_t::threads_per_worker;
  constexpr unsigned workers_per_block  = tuning.block_threads / threads_per_worker;
  if (static_cast<unsigned long long>(blockIdx.x) * workers_per_block
      >= partition.bin_offsets[num_kernel_bins] - partition.bin_offsets[0])
  {
    return;
  }
  // Convert per-bin segment counts into worker counts; each block builds its own prefix.
  __shared__ unsigned bin_work_offsets[num_kernel_bins + 1];
  if (threadIdx.x == 0)
  {
    unsigned prefix     = 0;
    bin_work_offsets[0] = 0;
    for (int bin_idx = 0; bin_idx < num_kernel_bins; ++bin_idx)
    {
      const auto config                  = lookup_bin_config<typename implementation_t::expansion>(bin_idx);
      const unsigned segments_per_worker = threads_per_worker / config.threads_per_segment;
      const unsigned num_segments_in_bin = partition.bin_offsets[bin_idx + 1] - partition.bin_offsets[bin_idx];
      prefix += num_segments_in_bin / segments_per_worker + (num_segments_in_bin % segments_per_worker != 0);
      bin_work_offsets[bin_idx + 1] = prefix;
    }
  }
  __syncthreads();
  // Key storage used by the sorting worker, separate from the work prefix.
  extern __shared__ __align__(256) unsigned char dynamic_smem[];
  const unsigned worker_thread_idx = threadIdx.x % threads_per_worker;
  const auto first_work_id =
    static_cast<unsigned long long>(blockIdx.x) * workers_per_block + threadIdx.x / threads_per_worker;
  const auto work_stride = static_cast<unsigned long long>(gridDim.x) * workers_per_block;
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
    const int bin_idx                  = first - 1;
    const auto config                  = lookup_bin_config<typename implementation_t::expansion>(bin_idx);
    const unsigned segments_per_worker = threads_per_worker / config.threads_per_segment;
    const auto segment_idx =
      partition.bin_offsets[bin_idx] + (work_id - bin_work_offsets[bin_idx]) * segments_per_worker
      + worker_thread_idx / config.threads_per_segment;
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

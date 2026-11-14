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

#include <cub/agent/agent_segmented_sort_lrb.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/detail/segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/__atomic/atomic.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__type_traits/conditional.h>

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
      bin_idx = Capacities::first_fit(end - begin);
      segment_rank_in_bin =
        ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>{bin_counts_shared[bin_idx]}.fetch_add(
          1u, ::cuda::memory_order_relaxed);
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < total_bins; i += static_cast<int>(blockDim.x))
  {
    if (bin_counts_shared[i])
    {
      bin_starts_shared[i] = ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>{counts_or_cursors[i]}.fetch_add(
        bin_counts_shared[i], ::cuda::memory_order_relaxed);
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

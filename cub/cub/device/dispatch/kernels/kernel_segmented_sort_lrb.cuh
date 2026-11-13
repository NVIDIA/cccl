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
#include <cub/detail/segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__atomic/atomic.h>
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
__launch_bounds__(current_policy<PolicySelector>().block_threads) _CCCL_KERNEL_ATTRIBUTES void partition_kernel(
  BeginIterator begins,
  EndIterator ends,
  unsigned num_segments,
  unsigned* counts_or_cursors,
  ::cuda::std::conditional_t<Scatter, unsigned, const unsigned>* ids,
  unsigned* offsets,
  unsigned* cursors,
  unsigned* blocks_completed)
{
  constexpr int total_bins = Capacities::size + 1;
  __shared__ unsigned bin_counts_shared[total_bins];
  [[maybe_unused]] __shared__ unsigned bin_starts_shared[total_bins];
  for (int bin_idx = static_cast<int>(threadIdx.x); bin_idx < total_bins; bin_idx += static_cast<int>(blockDim.x))
  {
    bin_counts_shared[bin_idx] = 0;
  }
  __syncthreads();
  const auto seg_id = (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
  int bin_idx       = -1;
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
  // Publish this block's counts only after every thread has finished its global updates.
  __syncthreads();
  if constexpr (Scatter)
  {
    if (bin_idx >= 0)
    {
      ids[bin_starts_shared[bin_idx] + segment_rank_in_bin] = static_cast<unsigned>(seg_id);
    }
  }
  else if (threadIdx.x == 0)
  {
    auto completed = ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>{*blocks_completed};
    // The last block to finish acquires all preceding counts; no block waits for another.
    if (completed.fetch_add(1, ::cuda::std::memory_order_acq_rel) == gridDim.x - 1)
    {
      unsigned prefix = 0;
      for (int bin_idx = 0; bin_idx < total_bins; ++bin_idx)
      {
        offsets[bin_idx] = cursors[bin_idx] = prefix;
        prefix += counts_or_cursors[bin_idx];
      }
      offsets[total_bins] = prefix;
    }
  }
}

// Consume this kernel's bins, packing segments into groups and invoking its sorting implementation.
// The kernel type carries the selector; launch bounds and the body select tuning for the compiled target.
template <typename Kernel, typename Key, typename BeginIterator, typename EndIterator, bool Descending = false>
__launch_bounds__(current_policy<typename Kernel::policy_selector>().block_threads,
                  current_policy<typename Kernel::policy_selector>().min_blocks_per_sm)
  _CCCL_KERNEL_ATTRIBUTES void sort_kernel(
    partition_result partition, BeginIterator begins, EndIterator ends, const Key* in, Key* out)
{
  using implementation_t = typename Kernel::template implementation_t<
    device_policy_getter<typename Kernel::policy_selector, current_tuning_cc().get()>>;
  using agent = agent_t<implementation_t>;
  __shared__ typename agent::TempStorage temp_storage;
  extern __shared__ __align__(256) unsigned char dynamic_smem[];
  agent::template process<Descending>(
    partition, begins, ends, in, out, temp_storage, reinterpret_cast<Key*>(dynamic_smem));
}
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END

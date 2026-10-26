// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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

#include <cub/detail/segmented_sort_capacity.cuh>
#include <cub/util_arch.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Count nonempty segments by capacity, or scatter their IDs using reserved per-bin ranges.
// Partition owns ID ranges only; it has no sorting configuration or key pointers.
template <typename Capacities, bool Scatter, typename BeginIterator, typename EndIterator>
__launch_bounds__(256) _CCCL_KERNEL_ATTRIBUTES void partition_kernel(
  BeginIterator begins, EndIterator ends, unsigned num_segments, unsigned* counts_or_cursors, unsigned* ids)
{
  __shared__ unsigned counts[Capacities::size + 1];
  [[maybe_unused]] __shared__ unsigned starts[Capacities::size + 1]; // Read only in the scatter pass.
  for (int bin = threadIdx.x; bin <= Capacities::size; bin += blockDim.x)
  {
    counts[bin] = 0;
  }
  __syncthreads();
  const unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
  int bin                = -1;
  unsigned rank          = 0;
  if (segment < num_segments)
  {
    const auto begin = begins[segment];
    const auto end   = ends[segment];
    if (end > begin)
    {
      bin  = Capacities::first_fit(end - begin);
      rank = atomicAdd(counts + bin, 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i <= Capacities::size; i += blockDim.x)
  {
    if (counts[i])
    {
      starts[i] = atomicAdd(counts_or_cursors + i, counts[i]);
    }
  }
  if constexpr (Scatter)
  {
    __syncthreads();
    if (bin >= 0)
    {
      ids[starts[bin] + rank] = segment;
    }
  }
}

// One thread converts bin counts into global ID offsets and initial scatter cursors.
template <typename Capacities>
_CCCL_KERNEL_ATTRIBUTES void segment_prefix_kernel(const unsigned* counts, unsigned* offsets, unsigned* cursors)
{
  unsigned prefix = 0;
  for (int bin = 0; bin <= Capacities::size; ++bin)
  {
    offsets[bin] = cursors[bin] = prefix;
    prefix += counts[bin];
  }
  offsets[Capacities::size + 1] = prefix;
}

// Consume one entry's bins, packing segments into work units and invoking its sorting worker.
template <typename Entry, typename Key, typename BeginIterator, typename EndIterator>
__launch_bounds__(Entry::tuning.block_threads, Entry::tuning.min_blocks_per_sm)
  _CCCL_KERNEL_ATTRIBUTES void sort_entry_kernel(
    partition_result bins, BeginIterator begins, EndIterator ends, const Key* in, Key* out)
{
  constexpr int bin_count  = Entry::capacities::size;
  constexpr unsigned units = Entry::tuning.block_threads / Entry::unit_threads;
  if (static_cast<unsigned long long>(blockIdx.x) * units >= bins.bin_offsets[bin_count] - bins.bin_offsets[0])
  {
    return;
  }
  // Convert segment counts into work-unit counts; each block builds its own prefix.
  __shared__ unsigned work_offsets[bin_count + 1];
  if (threadIdx.x == 0)
  {
    unsigned prefix = 0;
    work_offsets[0] = 0;
    for (int bin = 0; bin < bin_count; ++bin)
    {
      const auto config      = Entry::config(bin);
      const unsigned packing = Entry::unit_threads / config.threads;
      const unsigned count   = bins.bin_offsets[bin + 1] - bins.bin_offsets[bin];
      prefix += count / packing + (count % packing != 0);
      work_offsets[bin + 1] = prefix;
    }
  }
  __syncthreads();
  // Separate from the work prefix: key storage used by the sorting worker.
  extern __shared__ __align__(256) unsigned char scratch[];
  const unsigned lane = threadIdx.x % Entry::unit_threads;
  const auto initial  = static_cast<unsigned long long>(blockIdx.x) * units + threadIdx.x / Entry::unit_threads;
  const auto stride   = static_cast<unsigned long long>(gridDim.x) * units;
  for (auto work = initial; work < work_offsets[bin_count]; work += stride)
  {
    // Locate the bin containing this work index in the shared-memory prefix.
    int first = 1;
    int last  = bin_count;
    while (first < last)
    {
      const int mid = first + (last - first) / 2;
      if (work >= work_offsets[mid])
      {
        first = mid + 1;
      }
      else
      {
        last = mid;
      }
    }
    const int bin          = first - 1;
    const auto config      = Entry::config(bin);
    const unsigned packing = Entry::unit_threads / config.threads;
    const auto slot        = bins.bin_offsets[bin] + (work - work_offsets[bin]) * packing + lane / config.threads;
    Entry::implementation::invoke(
      slot < bins.bin_offsets[bin + 1] ? slot : ~0u,
      config,
      bins,
      begins,
      ends,
      in,
      out,
      reinterpret_cast<Key*>(scratch));
  }
}
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END

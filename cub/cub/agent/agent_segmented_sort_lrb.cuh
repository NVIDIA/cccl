// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Schedule capacity-binned segments and invoke the selected cooperative sorting worker.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/segmented_sort_lrb.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{
// Schedule a sorting entry's bins independently of its concrete segment-sorting implementation.
template <typename Implementation>
struct agent_t
{
  using implementation_t = Implementation;
  using capacities_t     = typename implementation_t::expansion::capacities;

  struct TempStorage
  {
    unsigned bin_work_offsets[capacities_t::size + 1];
  };

  template <bool Descending = false, typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void process(
    partition_result partition,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    TempStorage& temp_storage,
    Key* keys_shared)
  {
    constexpr auto tuning                = implementation_t::policy;
    constexpr int num_kernel_bins        = capacities_t::size;
    constexpr unsigned threads_per_group = implementation_t::threads_per_group;
    constexpr unsigned groups_per_block  = tuning.threads_per_block / threads_per_group;
    if (static_cast<unsigned long long>(blockIdx.x) * groups_per_block
        >= partition.bin_offsets[num_kernel_bins] - partition.bin_offsets[0])
    {
      return;
    }
    // Convert per-bin segment counts into work-item counts; each block builds its own prefix.
    auto& bin_work_offsets = temp_storage.bin_work_offsets;
    if (threadIdx.x == 0)
    {
      unsigned prefix     = 0;
      bin_work_offsets[0] = 0;
      for (int bin_idx = 0; bin_idx < num_kernel_bins; ++bin_idx)
      {
        const auto config                  = lookup_bin_config<typename implementation_t::expansion>(bin_idx);
        const unsigned segments_per_group  = threads_per_group / config.threads_per_segment;
        const unsigned num_segments_in_bin = partition.bin_offsets[bin_idx + 1] - partition.bin_offsets[bin_idx];
        prefix += (num_segments_in_bin / segments_per_group) + (num_segments_in_bin % segments_per_group != 0);
        bin_work_offsets[bin_idx + 1] = prefix;
      }
    }
    __syncthreads();
    const unsigned thread_idx_in_group = threadIdx.x % threads_per_group;
    const auto first_work_id =
      (static_cast<unsigned long long>(blockIdx.x) * groups_per_block) + (threadIdx.x / threads_per_group);
    const auto work_stride = static_cast<unsigned long long>(gridDim.x) * groups_per_block;
    // Each work item assigns one group up to segments_per_group segments from one bin.
    for (auto work_id = first_work_id; work_id < bin_work_offsets[num_kernel_bins]; work_id += work_stride)
    {
      // Locate this work ID's bin in the shared-memory prefix.
      int first = 1;
      int last  = num_kernel_bins;
      while (first < last)
      {
        const int mid = first + ((last - first) / 2);
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
      const auto segment_idx =
        partition.bin_offsets[bin_idx] + ((work_id - bin_work_offsets[bin_idx]) * segments_per_group)
        + (thread_idx_in_group / config.threads_per_segment);
      implementation_t::template invoke<Descending>(
        segment_idx < partition.bin_offsets[bin_idx + 1] ? segment_idx : ~0u,
        config,
        partition,
        begins,
        ends,
        in,
        out,
        keys_shared);
    }
  }
};
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END

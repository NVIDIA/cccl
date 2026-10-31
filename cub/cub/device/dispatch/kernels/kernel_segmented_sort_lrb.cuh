// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
#include <cub/util_arch.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{

template <typename Policy, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename CountT>
__launch_bounds__(Policy::block_threads)
  _CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedSortLrbCountKernel(
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    CountT* d_warp_counts,
    CountT* d_small_block_counts,
    CountT* d_large_block_counts,
    CountT* d_overflow_counts)
{
  using AgentT = AgentSegmentedSortLrbCount<Policy, BeginOffsetIteratorT, EndOffsetIteratorT, CountT>;
  __shared__ typename AgentT::TempStorage storage;

  lrb_histograms<Policy, CountT> global_hist{
    d_warp_counts, d_small_block_counts, d_large_block_counts, d_overflow_counts};

  AgentT agent(storage, d_begin_offsets, d_end_offsets, num_segments, global_hist);
  agent.Process();
}

template <typename Policy, typename CountT>
_CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedSortLrbPrefixKernel(
  CountT* d_warp_counts,
  CountT* d_small_block_counts,
  CountT* d_large_block_counts,
  CountT* d_overflow_counts,
  CountT* d_warp_offsets,
  CountT* d_small_block_offsets,
  CountT* d_large_block_offsets,
  CountT* d_overflow_offsets,
  CountT* d_warp_offsets_atomic,
  CountT* d_small_block_offsets_atomic,
  CountT* d_large_block_offsets_atomic,
  CountT* d_overflow_offsets_atomic,
  CountT* d_warp_group_offsets,
  CountT* d_small_block_group_offsets,
  CountT* d_large_block_group_offsets,
  lrb_summary<CountT>* d_summary)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
  {
    return;
  }

  lrb_histograms<Policy, CountT> counts{
    d_warp_counts, d_small_block_counts, d_large_block_counts, d_overflow_counts};

  finalize_lrb_metadata<Policy>(
    counts,
    d_warp_offsets,
    d_small_block_offsets,
    d_large_block_offsets,
    d_overflow_offsets,
    d_warp_offsets_atomic,
    d_small_block_offsets_atomic,
    d_large_block_offsets_atomic,
    d_overflow_offsets_atomic,
    d_warp_group_offsets,
    d_small_block_group_offsets,
    d_large_block_group_offsets,
    d_summary);
}

template <typename Policy, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename CountT>
__launch_bounds__(Policy::block_threads)
  _CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedSortLrbScatterKernel(
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    CountT* d_warp_offsets_atomic,
    CountT* d_small_block_offsets_atomic,
    CountT* d_large_block_offsets_atomic,
    CountT* d_overflow_offsets_atomic,
    CountT* d_segment_ids)
{
  using AgentT = AgentSegmentedSortLrbScatter<Policy, BeginOffsetIteratorT, EndOffsetIteratorT, CountT>;
  __shared__ typename AgentT::TempStorage storage;

  AgentT agent(
    storage,
    d_begin_offsets,
    d_end_offsets,
    num_segments,
    d_warp_offsets_atomic,
    d_small_block_offsets_atomic,
    d_large_block_offsets_atomic,
    d_overflow_offsets_atomic,
    d_segment_ids);
  agent.Process();
}

} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END

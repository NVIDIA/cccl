// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_scan.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename ActualInitValueT = typename InitValueT::value_type>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::segmented_scan_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void device_segmented_scan_kernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value)
{
  using segmented_scan_policy_t = typename ChainedPolicyT::ActivePolicy::segmented_scan_policy_t;

  using agent_segmented_scan_t = cub::detail::segmented_scan::agent_segmented_scan<
    segmented_scan_policy_t,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  __shared__ typename agent_segmented_scan_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  static constexpr int num_segments_per_block = segmented_scan_policy_t::segments_per_block;
  const auto work_id                          = num_segments_per_block * blockIdx.x;

  _CCCL_ASSERT(work_id < n_segments, "device_segmented_scan_kernel launch configuration results in access violation");

  if constexpr (num_segments_per_block == 1)
  {
    const OffsetT inp_begin_offset = begin_offset_d_in[work_id];
    const OffsetT inp_end_offset   = end_offset_d_in[work_id];
    const OffsetT out_begin_offset = begin_offset_d_out[work_id];

    agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
      .consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
  }
  else
  {
    OffsetT inp_begin_offsets[num_segments_per_block];
    OffsetT inp_end_offsets[num_segments_per_block];
    OffsetT out_begin_offsets[num_segments_per_block];

    if (work_id + num_segments_per_block - 1 < n_segments)
    {
#pragma unroll
      for (int i = 0; i < num_segments_per_block; ++i)
      {
        inp_begin_offsets[i] = begin_offset_d_in[work_id + i];
        inp_end_offsets[i]   = end_offset_d_in[work_id + i];
        out_begin_offsets[i] = begin_offset_d_out[work_id + i];
      }
      agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
        .consume_ranges(inp_begin_offsets, inp_end_offsets, out_begin_offsets);
    }
    else
    {
      int tail_size = n_segments - work_id;
      for (int i = 0; i < tail_size; ++i)
      {
        inp_begin_offsets[i] = begin_offset_d_in[work_id + i];
        inp_end_offsets[i]   = end_offset_d_in[work_id + i];
        out_begin_offsets[i] = begin_offset_d_out[work_id + i];
      }
      agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
        .consume_ranges(inp_begin_offsets, inp_end_offsets, out_begin_offsets, tail_size);
    }
  }
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END

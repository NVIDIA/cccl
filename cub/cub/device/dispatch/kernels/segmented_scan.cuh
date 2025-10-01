// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::SegmentedScanPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedScanKernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value)
{
  using SegmentedScanPolicyT = typename ChainedPolicyT::ActivePolicy::SegmentedScanPolicyT;

  // Define AgentSegmentedScanT
  using AgentSegmentedScanT = cub::detail::segmented_scan::AgentSegmentedScan<
    SegmentedScanPolicyT,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  // Declare shared memory of AgentSegmentedScanT::TempStorage type
  __shared__ typename AgentSegmentedScanT::TempStorage temp_storage;

  // Invoke agent logic
  ActualInitValueT _init_value = init_value;

  const auto segment_id = blockIdx.x;
  if (segment_id < n_segments)
  {
    OffsetT inp_begin_offset = begin_offset_d_in[segment_id];
    OffsetT inp_end_offset   = end_offset_d_in[segment_id];
    OffsetT out_begin_offset = begin_offset_d_out[segment_id];

    AgentSegmentedScanT(temp_storage, d_in, d_out, scan_op, _init_value)
      .ConsumeRange(inp_begin_offset, inp_end_offset, out_begin_offset);
  }
}

} // end of namespace detail::segmented_scan

CUB_NAMESPACE_END

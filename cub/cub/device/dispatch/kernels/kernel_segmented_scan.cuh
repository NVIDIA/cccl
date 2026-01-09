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

  const auto segment_id = blockIdx.x;

  _CCCL_ASSERT(segment_id < n_segments,
               "device_segmented_scan_kernel launch configuration results in access violation");

  const OffsetT inp_begin_offset = begin_offset_d_in[segment_id];
  const OffsetT inp_end_offset   = end_offset_d_in[segment_id];
  const OffsetT out_begin_offset = begin_offset_d_out[segment_id];

  agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
    .consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END

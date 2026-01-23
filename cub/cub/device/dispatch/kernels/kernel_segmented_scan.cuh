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
#include <cub/agent/agent_thread_segmented_scan.cuh>
#include <cub/agent/agent_warp_segmented_scan.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_type.cuh>

#include <cuda/iterator>

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
    InitValueT init_value,
    int num_segments_per_worker)
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
  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by block must be positive");
  _CCCL_ASSERT(num_segments_per_worker <= segmented_scan_policy_t::max_segments_per_block,
               "Requested number of segments to be processed by block exceeds compile-time maximum");

  const auto work_id = num_segments_per_worker * blockIdx.x;

  _CCCL_ASSERT(work_id < n_segments, "device_segmented_scan_kernel launch configuration results in access violation");

  if constexpr (segmented_scan_policy_t::max_segments_per_block == 1)
  {
    const OffsetT inp_begin_offset = begin_offset_d_in[work_id];
    const OffsetT inp_end_offset   = end_offset_d_in[work_id];
    const OffsetT out_begin_offset = begin_offset_d_out[work_id];

    agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
      .consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
  }
  else
  {
    const auto start_id = work_id;
    const auto end_id   = ::cuda::std::min<decltype(start_id)>(start_id + num_segments_per_worker, n_segments);
    int size            = end_id - start_id;

    auto worker_beg_off_d_in  = begin_offset_d_in + start_id;
    auto worker_end_off_d_in  = end_offset_d_in + start_id;
    auto worker_beg_off_d_out = begin_offset_d_out + start_id;

    agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
      .consume_ranges(worker_beg_off_d_in, worker_end_off_d_in, worker_beg_off_d_out, size);
  }
}

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
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::warp_segmented_scan_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void device_warp_segmented_scan_kernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    int /* num_segments_per_worker */)
{
  using policy_t = typename ChainedPolicyT::ActivePolicy::warp_segmented_scan_policy_t;

  using agent_t = cub::detail::segmented_scan::agent_warp_segmented_scan<
    policy_t,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  __shared__ typename agent_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  static constexpr int num_segments_per_warp   = policy_t::segments_per_warp;
  static constexpr unsigned int warps_in_block = int(policy_t::BLOCK_THREADS) >> cub::detail::log2_warp_threads;
  const unsigned int warp_id                   = threadIdx.x >> cub::detail::log2_warp_threads;

  const auto work_id = num_segments_per_warp * (blockIdx.x * warps_in_block) + warp_id;

  if constexpr (num_segments_per_warp == 1)
  {
    if (work_id < n_segments)
    {
      const OffsetT inp_begin_offset = begin_offset_d_in[work_id];
      const OffsetT inp_end_offset   = end_offset_d_in[work_id];
      const OffsetT out_begin_offset = begin_offset_d_out[work_id];

      agent_t(temp_storage, d_in, d_out, scan_op, _init_value)
        .consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
    }
  }
  else
  {
    OffsetT inp_end_offsets[num_segments_per_warp] = {};

    const ::cuda::strided_iterator<BeginOffsetIteratorInputT> raked_begin_inp{
      begin_offset_d_in + work_id, warps_in_block};
    const ::cuda::strided_iterator<BeginOffsetIteratorOutputT> raked_begin_out{
      begin_offset_d_out + work_id, warps_in_block};

    using span_t = ::cuda::std::span<OffsetT, num_segments_per_warp>;

    if (work_id + num_segments_per_warp * warps_in_block < n_segments)
    {
#pragma unroll
      for (int i = 0; i < num_segments_per_warp; ++i)
      {
        inp_end_offsets[i] = end_offset_d_in[work_id + i * warps_in_block];
      }
      agent_t(temp_storage, d_in, d_out, scan_op, _init_value)
        .consume_ranges(raked_begin_inp, span_t{inp_end_offsets}, raked_begin_out);
    }
    else
    {
      if (work_id < n_segments)
      {
        int tail_size = (n_segments - work_id) / warps_in_block;
        for (int i = 0; i < tail_size; ++i)
        {
          inp_end_offsets[i] = end_offset_d_in[work_id + i * warps_in_block];
        }
        agent_t(temp_storage, d_in, d_out, scan_op, _init_value)
          .consume_ranges(raked_begin_inp, span_t{inp_end_offsets}, raked_begin_out, tail_size);
      }
    }
  }
}

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
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::warp_segmented_scan_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void device_thread_segmented_scan_kernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    int num_segments_per_worker)
{
  using policy_t = typename ChainedPolicyT::ActivePolicy::thread_segmented_scan_policy_t;

  using agent_t = cub::detail::segmented_scan::agent_thread_segmented_scan<
    policy_t,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  __shared__ typename agent_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  agent_t agent(
    temp_storage, d_in, d_out, begin_offset_d_in, end_offset_d_in, begin_offset_d_out, n_segments, scan_op, _init_value);
  agent.consume_range(num_segments_per_worker);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END

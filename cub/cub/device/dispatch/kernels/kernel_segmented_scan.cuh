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
  using policy_t = typename ChainedPolicyT::ActivePolicy::segmented_scan_policy_t;

  using agent_t = cub::detail::segmented_scan::agent_segmented_scan<
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

  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by block must be positive");
  _CCCL_ASSERT(num_segments_per_worker <= policy_t::max_segments_per_block,
               "Requested number of segments to be processed by block exceeds compile-time maximum");

  const auto work_id = num_segments_per_worker * blockIdx.x;

  agent_t agent(temp_storage, d_in, d_out, scan_op, _init_value);

  if constexpr (policy_t::max_segments_per_block == 1)
  {
    _CCCL_ASSERT(num_segments_per_worker == 1, "Inconsistent parameters in device_warp_segmented_scan_kernel");
    _CCCL_ASSERT(work_id < n_segments, "device_segmented_scan_kernel launch configuration results in access violation");

    const OffsetT inp_begin_offset = begin_offset_d_in[work_id];
    const OffsetT inp_end_offset   = end_offset_d_in[work_id];
    const OffsetT out_begin_offset = begin_offset_d_out[work_id];

    agent.consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
  }
  else
  {
    if (work_id >= n_segments)
    {
      return;
    }

    const auto start_offset         = work_id;
    const auto suggested_end_offset = start_offset + num_segments_per_worker;

    using IdT             = decltype(work_id);
    const auto end_offset = ::cuda::std::min<IdT>(suggested_end_offset, n_segments);
    int size              = end_offset - start_offset;

    auto worker_beg_off_d_in  = begin_offset_d_in + start_offset;
    auto worker_end_off_d_in  = end_offset_d_in + start_offset;
    auto worker_beg_off_d_out = begin_offset_d_out + start_offset;

    agent.consume_ranges(worker_beg_off_d_in, worker_end_off_d_in, worker_beg_off_d_out, size);
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
    int num_segments_per_worker)
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

  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by warp must be positive");
  _CCCL_ASSERT(num_segments_per_worker <= policy_t::max_segments_per_warp,
               "Requested number of segments to be processed by warp exceeds compile-time maximum");

  static constexpr unsigned int warps_in_block = int(policy_t::BLOCK_THREADS) >> cub::detail::log2_warp_threads;
  const unsigned int warp_id                   = threadIdx.x >> cub::detail::log2_warp_threads;

  const auto work_id = num_segments_per_worker * (blockIdx.x * warps_in_block) + warp_id;

  if (work_id >= n_segments)
  {
    return;
  }

  agent_t agent(temp_storage, d_in, d_out, scan_op, _init_value);

  if constexpr (policy_t::max_segments_per_warp == 1)
  {
    const OffsetT inp_begin_offset = begin_offset_d_in[work_id];
    const OffsetT inp_end_offset   = end_offset_d_in[work_id];
    const OffsetT out_begin_offset = begin_offset_d_out[work_id];

    agent.consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
  }
  else
  {
    // agent consumes interleaved segments, to improve CTA' memory access locality

    // agent accesses offset iterators with index: thread_work_id = chunk_id * worker_thread_count + lane_id;
    // for 0 <= chunk_id < ::cuda::ceil_div<unsigned>(n_segments, worker_thread_count)
    //
    //  total_offset = num_segments_per_worker * (blockIdx.x * warps_in_block) + warp_id +
    //      warps_in_block * thread_work_id;
    //
    using IdT                = decltype(work_id);
    const auto segment_count = static_cast<IdT>(n_segments);

    const int n_segments_per_warp =
      (work_id + num_segments_per_worker * warps_in_block < segment_count)
        ? num_segments_per_worker
        : ::cuda::ceil_div(segment_count - work_id, warps_in_block);

    const ::cuda::strided_iterator raked_begin_inp{begin_offset_d_in + work_id, warps_in_block};
    const ::cuda::strided_iterator raked_end_inp{end_offset_d_in + work_id, warps_in_block};
    const ::cuda::strided_iterator raked_begin_out{begin_offset_d_out + work_id, warps_in_block};

    agent.consume_ranges(raked_begin_inp, raked_end_inp, raked_begin_out, n_segments_per_warp);
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
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::thread_segmented_scan_policy_t::BLOCK_THREADS))
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

  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by thread must be positive");

  agent_t agent(
    temp_storage, d_in, d_out, begin_offset_d_in, end_offset_d_in, begin_offset_d_out, n_segments, scan_op, _init_value);
  agent.consume_range(num_segments_per_worker);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END

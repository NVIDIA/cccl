// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_type.cuh>

#include <cuda/iterator>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename PolicySelector,
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
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).block.block_threads))
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_scan_kernel(
    _CCCL_GRID_CONSTANT const InputIteratorT d_in,
    _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorInputT begin_offset_d_in,
    _CCCL_GRID_CONSTANT const EndOffsetIteratorInputT end_offset_d_in,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorOutputT begin_offset_d_out,
    _CCCL_GRID_CONSTANT const OffsetT n_segments,
    _CCCL_GRID_CONSTANT const ScanOpT scan_op,
    _CCCL_GRID_CONSTANT const InitValueT init_value,
    _CCCL_GRID_CONSTANT const int num_segments_per_worker)
{
  static constexpr auto policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).block;
  using policy_t               = agent_segmented_scan_policy_t<
                  policy.block_threads,
                  policy.items_per_thread,
                  policy.load_algorithm,
                  policy.load_modifier,
                  policy.store_algorithm,
                  policy.scan_algorithm,
                  policy.max_segments_per_block>;

  _CCCL_ASSERT(policy.block_threads == policy_t::block_threads, "Block policy threads mismatch");
  _CCCL_ASSERT(policy.items_per_thread == policy_t::items_per_thread, "Block policy items-per-thread mismatch");
  _CCCL_ASSERT(policy.max_segments_per_block == policy_t::max_segments_per_block, "Block policy max segments mismatch");

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
  _CCCL_ASSERT(num_segments_per_worker <= policy.max_segments_per_block,
               "Requested number of segments to be processed by block exceeds compile-time maximum");

  const auto work_id = num_segments_per_worker * blockIdx.x;

  agent_t agent(temp_storage, d_in, d_out, scan_op, _init_value);

  if constexpr (policy.max_segments_per_block == 1)
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

    if (size == 1)
    {
      agent.consume_range(worker_beg_off_d_in[0], worker_end_off_d_in[0], worker_beg_off_d_out[0]);
    }
    else
    {
      agent.consume_ranges(worker_beg_off_d_in, worker_end_off_d_in, worker_beg_off_d_out, size);
    }
  }
}

template <typename PolicySelector,
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
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).warp.block_threads))
  _CCCL_KERNEL_ATTRIBUTES void device_warp_segmented_scan_kernel(
    _CCCL_GRID_CONSTANT const InputIteratorT d_in,
    _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorInputT begin_offset_d_in,
    _CCCL_GRID_CONSTANT const EndOffsetIteratorInputT end_offset_d_in,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorOutputT begin_offset_d_out,
    _CCCL_GRID_CONSTANT const OffsetT n_segments,
    _CCCL_GRID_CONSTANT const ScanOpT scan_op,
    _CCCL_GRID_CONSTANT const InitValueT init_value,
    _CCCL_GRID_CONSTANT const int num_segments_per_worker)
{
  static constexpr auto policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).warp;
  using policy_t               = agent_warp_segmented_scan_policy_t<
                  policy.block_threads,
                  policy.items_per_thread,
                  policy.load_algorithm,
                  policy.load_modifier,
                  policy.store_algorithm,
                  policy.max_segments_per_warp>;

  _CCCL_ASSERT(policy.block_threads == policy_t::block_threads, "Warp policy threads mismatch");
  _CCCL_ASSERT(policy.items_per_thread == policy_t::items_per_thread, "Warp policy items-per-thread mismatch");
  _CCCL_ASSERT(policy.max_segments_per_warp == policy_t::max_segments_per_warp, "Warp policy max segments mismatch");

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
  _CCCL_ASSERT(num_segments_per_worker <= policy.max_segments_per_warp,
               "Requested number of segments to be processed by warp exceeds compile-time maximum");

  static constexpr unsigned int warps_in_block = int(policy.block_threads) >> cub::detail::log2_warp_threads;
  const unsigned int warp_id                   = threadIdx.x >> cub::detail::log2_warp_threads;

  const auto work_id = num_segments_per_worker * (blockIdx.x * warps_in_block) + warp_id;

  if (work_id >= n_segments)
  {
    return;
  }

  agent_t agent(temp_storage, d_in, d_out, scan_op, _init_value);

  if constexpr (policy.max_segments_per_warp == 1)
  {
    const OffsetT inp_begin_offset = begin_offset_d_in[work_id];
    const OffsetT inp_end_offset   = end_offset_d_in[work_id];
    const OffsetT out_begin_offset = begin_offset_d_out[work_id];

    agent.consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
  }
  else
  {
    // Agent consumes interleaved segments to improve CTA' memory access locality

    // agent accesses offset iterators with index: thread_work_id = chunk_id * worker_thread_count + lane_id;
    // for 0 <= chunk_id < ::cuda::ceil_div<unsigned>(n_segments, worker_thread_count)
    //
    //  total_offset = num_segments_per_worker * (blockIdx.x * warps_in_block) + warp_id +
    //      warps_in_block * thread_work_id;
    //
    using IdT                = decltype(work_id);
    const auto segment_count = static_cast<IdT>(n_segments);

    const int n_segments_per_warp =
      (work_id + (num_segments_per_worker - 1) * warps_in_block < segment_count)
        ? num_segments_per_worker
        : ::cuda::ceil_div(segment_count - work_id, warps_in_block);

    if (num_segments_per_worker == 1)
    {
      // The branch should be taken by all warps. Otherwise, since consume_range and consume_ranges methods
      // re-use shared memory using different logic, race condition arises for temporary storage in shared memory.

      if (n_segments_per_warp == 1)
      {
        // only those warps that do not read past end of segment iterators do the work
        agent.consume_range(begin_offset_d_in[work_id], end_offset_d_in[work_id], begin_offset_d_out[work_id]);
      }
    }
    else
    {
      const ::cuda::strided_iterator raked_begin_inp{begin_offset_d_in + work_id, warps_in_block};
      const ::cuda::strided_iterator raked_end_inp{end_offset_d_in + work_id, warps_in_block};
      const ::cuda::strided_iterator raked_begin_out{begin_offset_d_out + work_id, warps_in_block};
      agent.consume_ranges(raked_begin_inp, raked_end_inp, raked_begin_out, n_segments_per_warp);
    }
  }
}

template <typename PolicySelector,
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
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).thread.block_threads))
  _CCCL_KERNEL_ATTRIBUTES void device_thread_segmented_scan_kernel(
    _CCCL_GRID_CONSTANT const InputIteratorT d_in,
    _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorInputT begin_offset_d_in,
    _CCCL_GRID_CONSTANT const EndOffsetIteratorInputT end_offset_d_in,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorOutputT begin_offset_d_out,
    _CCCL_GRID_CONSTANT const OffsetT n_segments,
    _CCCL_GRID_CONSTANT const ScanOpT scan_op,
    _CCCL_GRID_CONSTANT const InitValueT init_value,
    _CCCL_GRID_CONSTANT const int num_segments_per_worker)
{
  static constexpr auto policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).thread;
  using policy_t =
    agent_thread_segmented_scan_policy_t<policy.block_threads, policy.items_per_thread, policy.load_modifier>;

  _CCCL_ASSERT(policy.block_threads == policy_t::block_threads, "Thread policy threads mismatch");
  _CCCL_ASSERT(policy.items_per_thread == policy_t::items_per_thread, "Thread policy items-per-thread mismatch");

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

  // Agent consumes interleaved segments to improve CTA' memory access locality

  agent_t agent(
    temp_storage, d_in, d_out, begin_offset_d_in, end_offset_d_in, begin_offset_d_out, n_segments, scan_op, _init_value);
  agent.consume_range(num_segments_per_worker);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::WarpReduceBatchedWspro provides WSPRO-based batched parallel reduction of items partitioned across a CUDA
//! thread warp.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_type.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/bit>
#include <cuda/cmath>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/array>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @brief WarpReduceBatchedWspro provides WSPRO (Warp Shuffle Parallel Reduction Optimization) based
//!        batched parallel reduction of items partitioned across a CUDA thread warp.
//!
//! @tparam T
//!   Data type being reduced
//!
//! @tparam Batches
//!   Number of arrays to reduce in batch
//!
//! @tparam LogicalWarpThreads
//!   Number of threads per logical warp (must be a power-of-two)
template <typename T, int Batches, int LogicalWarpThreads>
struct WarpReduceBatchedWspro
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(LogicalWarpThreads > 1 && LogicalWarpThreads <= warp_threads,
                "LogicalWarpThreads must be in the range [2, 32]");
  static_assert(Batches >= 1, "Batches must be >= 1");

  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr auto is_arch_warp = (LogicalWarpThreads == warp_threads);

  static constexpr auto max_out_per_thread = ::cuda::ceil_div(Batches, LogicalWarpThreads);

  /// Shared memory storage layout type
  using TempStorage = NullType;

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  int physical_lane_id;
  int logical_lane_id;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceBatchedWspro(TempStorage& /*temp_storage*/)
      : physical_lane_id(static_cast<int>(::cuda::ptx::get_sreg_laneid()))
      , logical_lane_id(is_arch_warp ? physical_lane_id : physical_lane_id % LogicalWarpThreads)
  {}

  //---------------------------------------------------------------------
  // Batched reductions
  //---------------------------------------------------------------------

  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Reduce(const InputT& inputs,
         OutputT& outputs,
         ReductionOp reduction_op,
         ::cuda::std::uint32_t lane_mask = ::cuda::device::lane_mask::all().value())
  {
    // Need writeable array as scratch space
    auto values = ::cuda::std::array<T, Batches>{};
#pragma unroll
    for (int i = 0; i < Batches; ++i)
    {
      values[i] = inputs[i];
    }

    ReduceInplace(values, reduction_op, lane_mask);

#pragma unroll
    for (int i = 0; i < max_out_per_thread; ++i)
    {
      const auto batch_idx = i * LogicalWarpThreads + logical_lane_id;
      if (batch_idx < Batches)
      {
        outputs[i] = values[i];
      }
    }
  }

  template <typename InputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ReduceInplace(
    InputT& inputs, ReductionOp reduction_op, ::cuda::std::uint32_t lane_mask = ::cuda::device::lane_mask::all().value())
  {
#if defined(_CCCL_ASSERT_DEVICE)
    const auto logical_warp_leader = ::cuda::round_down(physical_lane_id, LogicalWarpThreads);
    const auto logical_warp_mask   = ::cuda::bitmask(logical_warp_leader, LogicalWarpThreads);
#endif // _CCCL_ASSERT_DEVICE
    _CCCL_ASSERT((lane_mask & logical_warp_mask) == logical_warp_mask,
                 "lane_mask must be consistent for each logical warp");

#pragma unroll
    for (int stride_intra_reduce = 1; stride_intra_reduce < LogicalWarpThreads; stride_intra_reduce *= 2)
    {
      const auto stride_inter_reduce = 2 * stride_intra_reduce;
      const auto is_left_lane        = logical_lane_id % stride_inter_reduce < stride_intra_reduce;
#pragma unroll
      for (int i = 0; i < Batches; i += stride_inter_reduce)
      {
        auto left_value      = inputs[i];
        const auto right_idx = i + stride_intra_reduce;
        // Needed for Batches < LogicalWarpThreads case
        // Chose to redundantly operate on the last batch to avoid relying on default construction of T
        // Unrolling of the loops should make this a compile-time selection
        const auto safe_right_idx = right_idx < Batches ? right_idx : Batches - 1;
        auto right_value          = inputs[safe_right_idx];
        // Each left lane exchanges its right value against a right lane's left value
        if (is_left_lane)
        {
          ::cuda::std::swap(left_value, right_value);
        }
        left_value = ::cuda::device::warp_shuffle_xor(left_value, stride_intra_reduce, lane_mask);
        // While the current implementation is possibly faster, another conditional swap here would allow for
        // non-commutative reductions which might be useful for (segmented) scan operations.
        inputs[i] = reduction_op(left_value, right_value);
      }
    }
    // Make sure results are in the beginning of the array instead of strided
#pragma unroll
    for (int i = 1; i < max_out_per_thread; ++i)
    {
      if (i * LogicalWarpThreads < Batches)
      {
        inputs[i] = inputs[i * LogicalWarpThreads];
      }
    }
  }

  template <typename InputT, typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sum(
    const InputT& inputs, OutputT& outputs, ::cuda::std::uint32_t lane_mask = ::cuda::device::lane_mask::all().value())
  {
    Reduce(inputs, outputs, ::cuda::std::plus<>{}, lane_mask);
  }
};
} // namespace detail

CUB_NAMESPACE_END

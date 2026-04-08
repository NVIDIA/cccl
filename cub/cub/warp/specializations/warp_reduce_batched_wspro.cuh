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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__utility/static_for.h>
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
template <typename T, int Batches, int LogicalWarpThreads, bool SyncPhysicalWarp>
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
  int logical_warp_id;

  /// 32-thread physical warp member mask of logical warp
  ::cuda::std::uint32_t member_mask;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceBatchedWspro(TempStorage& /*temp_storage*/)
      : physical_lane_id(static_cast<int>(::cuda::ptx::get_sreg_laneid()))
      , logical_lane_id(is_arch_warp ? physical_lane_id : physical_lane_id % LogicalWarpThreads)
      , logical_warp_id(is_arch_warp ? 0 : (physical_lane_id / LogicalWarpThreads))
      , member_mask(SyncPhysicalWarp ? 0xFFFFFFFFu : cub::WarpMask<LogicalWarpThreads>(logical_warp_id))
  {}

  //---------------------------------------------------------------------
  // Batched reductions
  //---------------------------------------------------------------------

  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Reduce(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    // Need writeable array as scratch space
    auto values = ::cuda::std::array<T, Batches>{};
#pragma unroll
    for (int i = 0; i < Batches; ++i)
    {
      values[i] = inputs[i];
    }

    ReduceInplace(values, reduction_op);

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

  template <int LeftIdx, int StrideInterReduce = LogicalWarpThreads, typename InputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void RecurseReductionTree(InputT& inputs, ReductionOp reduction_op)
  {
    constexpr auto stride_intra_reduce = StrideInterReduce / 2;
    constexpr auto right_idx           = LeftIdx + stride_intra_reduce;
    constexpr auto base_case           = stride_intra_reduce == 1;
    if constexpr (!base_case)
    {
      // calculate left value
      RecurseReductionTree<LeftIdx, stride_intra_reduce, InputT, ReductionOp>(inputs, reduction_op);
      if constexpr (right_idx < Batches)
      {
        // calculate right value if it exists
        RecurseReductionTree<right_idx, stride_intra_reduce>(inputs, reduction_op);
      }
    }
    auto left_value = inputs[LeftIdx];
    // Needed for Batches < LogicalWarpThreads case
    // Chose to redundantly operate on the last batch to avoid relying on default construction of T
    constexpr auto safe_right_idx = right_idx < Batches ? right_idx : Batches - 1;
    auto right_value              = inputs[safe_right_idx];
    // Each left lane exchanges its right value against a right lane's left value
    const auto is_left_lane = logical_lane_id % StrideInterReduce < stride_intra_reduce;
    if (is_left_lane)
    {
      ::cuda::std::swap(left_value, right_value);
    }
    left_value = ::cuda::device::warp_shuffle_xor(left_value, stride_intra_reduce, member_mask);
    // While the current implementation is possibly faster, another conditional swap here would allow for
    // non-commutative reductions which might be useful for (segmented) scan operations.
    inputs[LeftIdx] = reduction_op(left_value, right_value);
  }

  template <typename InputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ReduceInplace(InputT& inputs, ReductionOp reduction_op)
  {
    ::cuda::static_for<0, max_out_per_thread>([&](auto out_idx) {
      constexpr auto offset = out_idx() * LogicalWarpThreads;
      RecurseReductionTree<offset>(inputs, reduction_op);
    });
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
};
} // namespace detail

CUB_NAMESPACE_END

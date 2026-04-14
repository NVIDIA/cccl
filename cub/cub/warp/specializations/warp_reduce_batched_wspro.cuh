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

  template <bool ToBlocked, typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Reduce(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    // Needed in case of outputs aliasing inputs
    ::cuda::std::array<T, max_out_per_thread> intermediate_outputs;

    ::cuda::static_for<0, max_out_per_thread>([&](auto out_idx) {
      constexpr auto first_idx      = out_idx * LogicalWarpThreads;
      intermediate_outputs[out_idx] = RecurseReductionTree<ToBlocked, first_idx>(inputs, reduction_op);
    });

    ::cuda::static_for<0, max_out_per_thread>([&](auto out_idx) {
      outputs[out_idx()] = intermediate_outputs[out_idx];
    });
  }

  template <bool ToBlocked, int BaseIdxStriped, int PrevStride = LogicalWarpThreads, typename InputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T RecurseReductionTree(const InputT& inputs, ReductionOp reduction_op)
  {
    // "Transpose" index
    constexpr auto base_idx_blocked =
      BaseIdxStriped / LogicalWarpThreads + (BaseIdxStriped % LogicalWarpThreads) * max_out_per_thread;
    constexpr auto base_idx = ToBlocked ? base_idx_blocked : BaseIdxStriped;
    // Important: By catching out-of-range indices at all levels instead of just at the base level we avoid unnecessary
    // shuffles.
    if constexpr (base_idx >= Batches)
    {
      // Dummy value needed when Batches is not an integer multiple of LogicalWarpThreads.
      // In that case some shuffles have no second batch to exchange against, so we exchange a dummy value.
      // Chose to use the last batch to avoid relying on default construction of T
      return inputs[Batches - 1];
    }
    else if constexpr (PrevStride == 1)
    {
      // Recursion base case
      return inputs[base_idx];
    }
    // Explicit "else" branch needed to avoid compiler error on "% 0".
    else
    {
      constexpr auto stride             = PrevStride / 2;
      constexpr auto offset_idx_striped = BaseIdxStriped + stride;

      auto exch_value = RecurseReductionTree<ToBlocked, BaseIdxStriped, stride>(inputs, reduction_op);
      auto keep_value = RecurseReductionTree<ToBlocked, offset_idx_striped, stride>(inputs, reduction_op);

      // Each left lane exchanges its offset value against a right lane's base value
      const auto is_left_lane = logical_lane_id % PrevStride < stride;
      if (is_left_lane)
      {
        ::cuda::std::swap(exch_value, keep_value);
      }
      exch_value = ::cuda::device::warp_shuffle_xor(exch_value, stride, member_mask);
      // While the current implementation is possibly faster, another conditional swap here would allow for
      // non-commutative reductions which might be useful for (segmented) scan operations.
      return reduction_op(exch_value, keep_value);
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END

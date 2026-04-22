// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::warp_reduce_batched_wspro provides WSPRO-based batched parallel reduction of items partitioned across a CUDA
//! warp.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_arch.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
// Next two for REDUX helpers
#include <cub/thread/thread_operators.cuh>
#include <cub/warp/specializations/warp_reduce_shfl.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__utility/static_for.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @brief warp_reduce_batched_wspro provides WSPRO (Warp Shuffle Parallel Reduction Optimization) based
//!        batched parallel reduction of items partitioned across a CUDA warp.
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
struct warp_reduce_batched_wspro
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");
  static_assert(Batches >= 0, "Batches must be >= 0");

  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  //! Whether the logical warp size and the physical warp size coincide
  static constexpr auto is_arch_warp = (LogicalWarpThreads == warp_threads);

  static constexpr auto max_out_per_thread = ::cuda::ceil_div(Batches, LogicalWarpThreads);

  //! Shared memory storage layout type
  using TempStorage = NullType;

  int physical_lane_id;
  int logical_lane_id;
  int logical_warp_id;

  ::cuda::std::uint32_t member_mask;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE warp_reduce_batched_wspro(TempStorage& /*temp_storage*/)
      : physical_lane_id(static_cast<int>(::cuda::ptx::get_sreg_laneid()))
      , logical_lane_id(is_arch_warp ? physical_lane_id : physical_lane_id % LogicalWarpThreads)
      , logical_warp_id(is_arch_warp ? 0 : (physical_lane_id / LogicalWarpThreads))
      , member_mask(SyncPhysicalWarp ? 0xFFFFFFFFu : cub::WarpMask<LogicalWarpThreads>(logical_warp_id))
  {}

  template <bool ToBlocked, typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Reduce(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    // Dispatch to more efficient intrinsics when applicable
    // For more batches, WSPRO gives significantly better throughput, while latency is only slightly better for
    // REDUX. For subwarps both throughput and latency are worse with REDUX independent of the number of batches. This
    // might be a codegen issue.
    constexpr bool redux_performs_better = is_arch_warp && Batches <= 8;
    if constexpr (is_redux_enabled_cuda_operator<ReductionOp, T> && redux_performs_better)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80, ({
                     ReduceRedux<ToBlocked>(inputs, outputs, reduction_op);
                     return;
                   }))
    }

    // Needed in case of outputs aliasing inputs
    ::cuda::std::array<T, max_out_per_thread> intermediate_outputs;

    ::cuda::static_for<0, max_out_per_thread>([&](auto out_idx) {
      constexpr auto first_idx      = out_idx * LogicalWarpThreads;
      intermediate_outputs[out_idx] = RecurseReductionTree<ToBlocked, first_idx>(inputs, reduction_op);
    });

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < max_out_per_thread; ++i)
    {
      outputs[i] = intermediate_outputs[i];
    }
  }

  template <bool ToBlocked, typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void ReduceRedux(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    // Needed to avoid compiler error on "/ 0" and improve compile time.
    if constexpr (Batches != 0)
    {
      // Needed in case of outputs aliasing inputs
      ::cuda::std::array<T, max_out_per_thread> intermediate_outputs;

      // Can't use the full member mask given SyncPhysicalWarp==true because it affects the result of the reduction.
      const auto reduce_mask = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Batches; ++i)
      {
        auto result         = cub::detail::reduce_op_sync(inputs[i], reduce_mask, reduction_op);
        const auto out_lane = ToBlocked ? i / max_out_per_thread : i % LogicalWarpThreads;
        const auto out_idx  = ToBlocked ? i % max_out_per_thread : i / LogicalWarpThreads;
        if (logical_lane_id == out_lane)
        {
          intermediate_outputs[out_idx] = result;
        }
      }

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < max_out_per_thread; ++i)
      {
        outputs[i] = intermediate_outputs[i];
      }
    }
  }

  template <bool ToBlocked, int BaseIdxStriped, int PrevStride = LogicalWarpThreads, typename InputT, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE T RecurseReductionTree(const InputT& inputs, ReductionOp reduction_op)
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
      // Chose to use the last batch to avoid relying on default construction of T.
      return inputs[Batches - 1];
    }
    else if constexpr (PrevStride == 1)
    {
      // Recursion base case
      return inputs[base_idx];
    }
    // Explicit "else" branch needed to avoid compiler error on "% 0" and improve compile time.
    else
    {
      constexpr auto stride             = PrevStride / 2;
      constexpr auto offset_idx_striped = BaseIdxStriped + stride;

      const auto base_value   = RecurseReductionTree<ToBlocked, BaseIdxStriped, stride>(inputs, reduction_op);
      const auto offset_value = RecurseReductionTree<ToBlocked, offset_idx_striped, stride>(inputs, reduction_op);

      // Each left lane exchanges its offset value against a right lane's base value
      const auto is_left_lane = (logical_lane_id % PrevStride) < stride;
      const auto keep_value   = is_left_lane ? base_value : offset_value;
      auto exch_value         = is_left_lane ? offset_value : base_value;
      exch_value              = ::cuda::device::warp_shuffle_xor(exch_value, stride, member_mask);
      // While the current implementation is possibly faster, another conditional swap here would allow for
      // non-commutative reductions which might be useful for (segmented) scan operations.
      return static_cast<T>(reduction_op(exch_value, keep_value));
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END

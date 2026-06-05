// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! @rst
//! The ``cub::WarpReduceBatchedBroadcast`` class provides :ref:`collective <collective-primitives>` methods for
//! performing batched warp-wide reductions whose aggregates are returned to every participating logical lane.
//! @endrst

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
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

//! @rst
//! Batched warp-wide reduction adapter that broadcasts every batch aggregate to every participating logical lane.
//! ``Sum`` and ``CommutativeReduce`` use a shuffle all-reduce fast path and require commutative
//! reduction operators. Use ``WarpReduceBatched`` directly when only owner lanes need the batch outputs.
//! @endrst
template <typename T, int Batches, int LogicalWarpThreads = detail::warp_threads, bool SyncPhysicalWarp = false>
class WarpReduceBatchedBroadcast
{
  static_assert(Batches > 0, "Batches must be greater than zero");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");

  template <typename InputType, typename OutputType, typename ReductionOp>
  static _CCCL_DEVICE_API _CCCL_FORCEINLINE void check_constraints()
  {
    static_assert(detail::is_fixed_size_random_access_range_v<InputType>,
                  "InputType must support operator[] and have a compile-time size");
    static_assert(detail::is_fixed_size_random_access_range_v<OutputType>,
                  "OutputType must support operator[] and have a compile-time size");
    static_assert(detail::static_size_v<InputType> == Batches, "Input size must match Batches");
    static_assert(detail::static_size_v<OutputType> == Batches, "Output size must match Batches");
    static_assert(detail::has_binary_call_operator<ReductionOp, T>::value,
                  "ReductionOp must have the binary call operator: operator(T, T)");
  }

  template <typename OutputType, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void commutative_all_reduce_batches(OutputType& outputs, ReductionOp reduction_op)
  {
    const auto lane_id         = cub::detail::logical_lane_id<LogicalWarpThreads>();
    const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const auto member_mask =
      SyncPhysicalWarp ? 0xFFFFFFFFu : static_cast<unsigned int>(cub::WarpMask<LogicalWarpThreads>(logical_warp_id));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = LogicalWarpThreads / 2; offset > 0; offset >>= 1)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int batch = 0; batch < Batches; ++batch)
      {
        const T peer   = cub::ShuffleIndex<LogicalWarpThreads>(outputs[batch], lane_id ^ offset, member_mask);
        outputs[batch] = reduction_op(outputs[batch], peer);
      }
    }
  }

public:
  /// @smemstorage{WarpReduceBatchedBroadcast}
  using TempStorage = cub::NullType;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit WarpReduceBatchedBroadcast(TempStorage& /*temp_storage*/) {}

  template <typename InputType, typename OutputType, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  CommutativeReduce(const InputType& inputs, OutputType& outputs, ReductionOp reduction_op)
  {
    check_constraints<InputType, OutputType, ReductionOp>();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      outputs[batch] = inputs[batch];
    }
    commutative_all_reduce_batches(outputs, reduction_op);
  }

  template <typename InputType, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::array<T, Batches>
  CommutativeReduce(const InputType& inputs, ReductionOp reduction_op)
  {
    ::cuda::std::array<T, Batches> outputs{};
    CommutativeReduce(inputs, outputs, reduction_op);
    return outputs;
  }

  template <typename InputType, typename OutputType>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Sum(const InputType& inputs, OutputType& outputs)
  {
    CommutativeReduce(inputs, outputs, ::cuda::std::plus<>{});
  }

  template <typename InputType>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::array<T, Batches> Sum(const InputType& inputs)
  {
    return CommutativeReduce(inputs, ::cuda::std::plus<>{});
  }
};

CUB_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! @rst
//! The ``cub::WarpReduceBroadcast`` class provides :ref:`collective <collective-primitives>` methods for
//! computing warp-wide reductions whose aggregate is returned to every participating logical lane.
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

#include <cub/thread/thread_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__functional/operations.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! Warp-wide reduction adapter that broadcasts the aggregate to every participating logical lane.
//! ``Sum`` uses a shuffle all-reduce fast path. Generic ``Reduce`` preserves CUB's non-commutative
//! reduction semantics by using the owner-lane result and broadcasting it.
//! @endrst
template <typename T, int LogicalWarpThreads = detail::warp_threads>
class WarpReduceBroadcast
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");

  using WarpReduceT = cub::WarpReduce<T, LogicalWarpThreads>;

  typename WarpReduceT::TempStorage& temp_storage;

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T commutative_all_reduce(T input, ReductionOp reduction_op) const
  {
    const auto lane_id         = cub::detail::logical_lane_id<LogicalWarpThreads>();
    const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = LogicalWarpThreads / 2; offset > 0; offset >>= 1)
    {
      const T peer = cub::ShuffleIndex<LogicalWarpThreads>(input, lane_id ^ offset, member_mask);
      input        = reduction_op(input, peer);
    }
    return input;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T broadcast_from_lane0(T aggregate) const
  {
    const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
    return cub::ShuffleIndex<LogicalWarpThreads>(aggregate, 0, member_mask);
  }

public:
  /// @smemstorage{WarpReduceBroadcast}
  using TempStorage = typename WarpReduceT::TempStorage;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit WarpReduceBroadcast(TempStorage& temp_storage)
      : temp_storage(temp_storage)
  {}

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    return commutative_all_reduce(input, ::cuda::std::plus<>{});
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputType>)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(const InputType& input)
  {
    return commutative_all_reduce(cub::ThreadReduce(input, ::cuda::std::plus<>{}), ::cuda::std::plus<>{});
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input, int valid_items)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage).Sum(input, valid_items));
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage).Reduce(input, reduction_op));
  }

  _CCCL_TEMPLATE(typename InputType, typename ReductionOp)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputType>)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(const InputType& input, ReductionOp reduction_op)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage).Reduce(input, reduction_op));
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int valid_items)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage).Reduce(input, reduction_op, valid_items));
  }
};

CUB_NAMESPACE_END

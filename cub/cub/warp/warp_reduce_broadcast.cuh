// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <nv/target>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``WarpReduceBroadcast`` class provides :ref:`collective <collective-primitives>` methods for computing
//! warp-wide reductions whose aggregate is returned to every participating logical lane.
//!
//! Overview
//! ++++++++
//!
//! - Supports "logical" warps smaller than the physical warp size (e.g., logical warps of 8 threads).
//! - Non-partial ``Sum``, ``Max``, and ``Min`` overloads use all-reduce fast paths for integral types.
//! - Integral all-reduce fast paths use a different reduction tree than ``WarpReduce`` for recognized commutative
//!   integral operators.
//! - Non-integral ``Sum`` preserves ``WarpReduce::Sum`` ordering by reducing to the owner lane and broadcasting the
//!   aggregate.
//! - Generic ``Reduce`` preserves ``WarpReduce`` ordering by reducing to the owner lane and broadcasting the aggregate,
//!   except recognized commutative integral operators use the integral all-reduce fast path.
//! - Segmented reduction overloads are intentionally omitted because a single broadcast source is ambiguous when the
//!   logical warp contains multiple segment aggregates. Use ``WarpReduce`` segmented reductions and explicit
//!   per-segment propagation when needed.
//! - Like all CUB warp collectives, every lane in each participating logical warp must invoke the same collective.
//!   Integral fast paths enforce this through the logical-warp mask passed to ``__reduce_*_sync``.
//! - Construction always follows the ``cub::WarpReduce`` temporary storage contract; the integral all-reduce fast paths
//!   do not read that storage, so the normal ``@smemwarpreuse`` guidance is conservative for those paths.
//! - This primitive is useful for kernels that need the aggregate in every lane and would otherwise call
//!   ``WarpReduce`` followed by an explicit warp broadcast.
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - Uses synchronization-free warp shuffle instructions when applicable.
//! - Uses a single hardware warp all-reduce instruction on SM80 and newer for supported integral all-reduce overloads
//!   when the reduced value type is no wider than ``unsigned int``. Multi-item overloads reduce ``ThreadReduce``'s
//!   accumulator type before converting to ``T``.
//! - Reuses ``WarpReduce`` for partial reductions, non-integral ``Sum``, ``Max``, and ``Min`` reductions, and generic
//!   ``Reduce`` operations with unrecognized or non-commutative operators.
//!
//! @warpcollective{WarpReduceBroadcast}
//!
//! Simple Example
//! ++++++++++++++
//!
//! The code snippet below illustrates a sum reduction within a single 32-thread warp, with the aggregate returned to
//! every lane:
//!
//! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_broadcast_api.cu
//!     :language: c++
//!     :dedent:
//!     :start-after: example-begin warp-reduce-broadcast-overview
//!     :end-before: example-end warp-reduce-broadcast-overview
//!
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type.
//!
//! @tparam LogicalWarpThreads
//!   **[optional]** The number of threads per "logical" warp (may be less than the number of hardware warp threads but
//!   has to be a power of two). Default is the warp size of the targeted CUDA compute-capability (e.g., 32 threads for
//!   SM80).
template <typename T, int LogicalWarpThreads = detail::warp_threads>
class WarpReduceBroadcast
{
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");

  using WarpReduceT = WarpReduce<T, LogicalWarpThreads>;

  //! Shared storage reference.
  typename WarpReduceT::TempStorage& temp_storage_;

  template <typename ReductionT, typename ReductionOp>
  static constexpr bool is_commutative_associative_integral_reduction_v =
    ::cuda::std::is_integral_v<ReductionT>
    && (detail::is_cuda_minimum_maximum_v<::cuda::std::remove_cvref_t<ReductionOp>, ReductionT>
        || detail::is_cuda_std_plus_v<::cuda::std::remove_cvref_t<ReductionOp>, ReductionT>
        || detail::is_cuda_std_bitwise_v<::cuda::std::remove_cvref_t<ReductionOp>, ReductionT>);

  template <typename ReductionT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ReductionT
  commutative_associative_all_reduce(ReductionT input, ReductionOp reduction_op) const
  {
    // This tree is intentionally different from WarpReduce's owner-lane tree. Use it only for operations where
    // reordering and reassociation cannot change the value.
    if constexpr (LogicalWarpThreads == 1)
    {
      return input;
    }
    else
    {
      const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
      const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int offset = LogicalWarpThreads / 2; offset > 0; offset >>= 1)
      {
        const ReductionT peer = ::cuda::device::warp_shuffle_xor<LogicalWarpThreads>(input, offset, member_mask);
        input                 = reduction_op(input, peer);
      }

      return input;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T broadcast_from_lane0(T aggregate) const
  {
    if constexpr (LogicalWarpThreads == 1)
    {
      return aggregate;
    }
    else
    {
      const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
      const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
      return cub::ShuffleIndex<LogicalWarpThreads>(aggregate, 0, member_mask);
    }
  }

  template <typename ReductionT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ReductionT
  integral_all_reduce(ReductionT input, ReductionOp reduction_op) const
  {
    static_assert(::cuda::std::is_integral_v<ReductionT>, "ReductionT must be an integral type");

    if constexpr (LogicalWarpThreads == 1)
    {
      return input;
    }
    else if constexpr (sizeof(ReductionT) <= sizeof(unsigned))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80,
                   (const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
                    const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
                    return cub::detail::reduce_op_sync(input, member_mask, reduction_op);))
    }

    return commutative_associative_all_reduce(input, reduction_op);
  }

  template <typename InputT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
  integral_all_reduce_items(const InputT& input, ReductionOp reduction_op) const
  {
    const auto thread_reduction    = cub::ThreadReduce(input, reduction_op);
    const auto converted_reduction = static_cast<T>(thread_reduction);
    return integral_all_reduce(converted_reduction, reduction_op);
  }

public:
  //! \smemstorage{WarpReduceBroadcast}
  using TempStorage = typename WarpReduceT::TempStorage;

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Logical warp and lane identifiers are constructed from the calling thread's lane id.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpReduceBroadcast(TempStorage& temp_storage)
      : temp_storage_(temp_storage)
  {}

  //! @}
  //! @name Sum reductions
  //! @{

  //! @rst
  //! Computes a warp-wide sum of one item per lane and returns the aggregate to every participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! The integral fast path does not read ``TempStorage``, so the ``@smemwarpreuse`` restriction does not apply when it
  //! is selected. It uses a different reduction tree than ``WarpReduce::Sum`` for commutative integral operators.
  //! Non-integral types use ``WarpReduce`` temporary storage to preserve ``WarpReduce::Sum`` ordering.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @return
  //!   The warp-wide sum.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return integral_all_reduce(input, ::cuda::std::plus<>{});
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Sum(input));
    }
  }

  //! @rst
  //! Computes a warp-wide sum of multiple items per lane and returns the aggregate to every participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! The integral fast path does not read ``TempStorage``, so the ``@smemwarpreuse`` restriction does not apply when it
  //! is selected. It uses a different reduction tree than ``WarpReduce::Sum`` for commutative integral operators.
  //! Multi-item overloads first reduce this lane's items with ``ThreadReduce``, convert that partial aggregate to
  //! ``T``, and all-reduce in ``T``. Non-integral types use ``WarpReduce`` temporary storage to preserve
  //! ``WarpReduce::Sum`` ordering.
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @param[in] input
  //!   Calling lane's input items.
  //!
  //! @return
  //!   The warp-wide sum.
  _CCCL_TEMPLATE(typename InputT)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputT>)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(const InputT& input)
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return integral_all_reduce_items(input, ::cuda::std::plus<>{});
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Sum(input));
    }
  }

  //! @rst
  //! Computes a warp-wide sum of up to ``valid_items`` items and returns the aggregate to every participating logical
  //! lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! Partial reductions use ``WarpReduce`` temporary storage; the integral fast path is not used.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @param[in] valid_items
  //!   Number of valid items across the logical warp. Must be in the range ``[1, LogicalWarpThreads]``.
  //!
  //! @return
  //!   The sum of the valid items.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input, int valid_items)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage_).Sum(input, valid_items));
  }

  //! @}
  //! @name Minimum and maximum reductions
  //! @{

  //! @rst
  //! Computes a warp-wide maximum of one item per lane and returns the aggregate to every participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! The integral fast path does not read ``TempStorage``, so the ``@smemwarpreuse`` restriction does not apply when it
  //! is selected. It uses a different reduction tree than ``WarpReduce::Max`` for commutative integral operators.
  //! Non-integral types use ``WarpReduce`` temporary storage.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @return
  //!   The warp-wide maximum.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Max(T input)
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return integral_all_reduce(input, ::cuda::maximum<>{});
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Max(input));
    }
  }

  //! @rst
  //! Computes a warp-wide maximum of multiple items per lane and returns the aggregate to every participating logical
  //! lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! The integral fast path does not read ``TempStorage``, so the ``@smemwarpreuse`` restriction does not apply when it
  //! is selected. It uses a different reduction tree than ``WarpReduce::Max`` for commutative integral operators.
  //! Multi-item overloads first reduce this lane's items with ``ThreadReduce``, convert that partial aggregate to
  //! ``T``, and all-reduce in ``T``. Non-integral types use ``WarpReduce`` temporary storage.
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @param[in] input
  //!   Calling lane's input items.
  //!
  //! @return
  //!   The warp-wide maximum.
  _CCCL_TEMPLATE(typename InputT)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputT>)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Max(const InputT& input)
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return integral_all_reduce_items(input, ::cuda::maximum<>{});
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Max(input));
    }
  }

  //! @rst
  //! Computes a warp-wide maximum of up to ``valid_items`` items and returns the aggregate to every participating
  //! logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! Partial reductions use ``WarpReduce`` temporary storage; the integral fast path is not used.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @param[in] valid_items
  //!   Number of valid items across the logical warp. Must be in the range ``[1, LogicalWarpThreads]``.
  //!
  //! @return
  //!   The maximum of the valid items.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Max(T input, int valid_items)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage_).Max(input, valid_items));
  }

  //! @rst
  //! Computes a warp-wide minimum of one item per lane and returns the aggregate to every participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! The integral fast path does not read ``TempStorage``, so the ``@smemwarpreuse`` restriction does not apply when it
  //! is selected. It uses a different reduction tree than ``WarpReduce::Min`` for commutative integral operators.
  //! Non-integral types use ``WarpReduce`` temporary storage.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @return
  //!   The warp-wide minimum.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Min(T input)
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return integral_all_reduce(input, ::cuda::minimum<>{});
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Min(input));
    }
  }

  //! @rst
  //! Computes a warp-wide minimum of multiple items per lane and returns the aggregate to every participating logical
  //! lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! The integral fast path does not read ``TempStorage``, so the ``@smemwarpreuse`` restriction does not apply when it
  //! is selected. It uses a different reduction tree than ``WarpReduce::Min`` for commutative integral operators.
  //! Multi-item overloads first reduce this lane's items with ``ThreadReduce``, convert that partial aggregate to
  //! ``T``, and all-reduce in ``T``. Non-integral types use ``WarpReduce`` temporary storage.
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @param[in] input
  //!   Calling lane's input items.
  //!
  //! @return
  //!   The warp-wide minimum.
  _CCCL_TEMPLATE(typename InputT)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputT>)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Min(const InputT& input)
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return integral_all_reduce_items(input, ::cuda::minimum<>{});
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Min(input));
    }
  }

  //! @rst
  //! Computes a warp-wide minimum of up to ``valid_items`` items and returns the aggregate to every participating
  //! logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! Partial reductions use ``WarpReduce`` temporary storage; the integral fast path is not used.
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @param[in] valid_items
  //!   Number of valid items across the logical warp. Must be in the range ``[1, LogicalWarpThreads]``.
  //!
  //! @return
  //!   The minimum of the valid items.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Min(T input, int valid_items)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage_).Min(input, valid_items));
  }

  //! @}
  //! @name Generic reductions
  //! @{

  //! @rst
  //! Computes a warp-wide reduction using the specified binary reduction functor and returns the aggregate to every
  //! participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! Recognized commutative integral operators use the integral all-reduce fast path, do not read ``TempStorage``, and
  //! use a different reduction tree than ``WarpReduce::Reduce``. The ``@smemwarpreuse`` restriction does not apply when
  //! that fast path is selected. Other operators use ``WarpReduce`` temporary storage to preserve
  //! ``WarpReduce::Reduce`` ordering.
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator.
  //!
  //! @return
  //!   The warp-wide reduction aggregate.
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    if constexpr (is_commutative_associative_integral_reduction_v<T, ReductionOp>)
    {
      return integral_all_reduce(input, reduction_op);
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Reduce(input, reduction_op));
    }
  }

  //! @rst
  //! Computes a warp-wide reduction of multiple items per lane and returns the aggregate to every participating logical
  //! lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! Recognized commutative integral operators use the integral all-reduce fast path, do not read ``TempStorage``, and
  //! use a different reduction tree than ``WarpReduce::Reduce``. The ``@smemwarpreuse`` restriction does not apply when
  //! that fast path is selected. Multi-item overloads first reduce this lane's items with ``ThreadReduce``, convert
  //! that partial aggregate to ``T``, and all-reduce in ``T``. Other operators use ``WarpReduce`` temporary storage to
  //! preserve ``WarpReduce::Reduce`` ordering.
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] input
  //!   Calling lane's input items.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator.
  //!
  //! @return
  //!   The warp-wide reduction aggregate.
  _CCCL_TEMPLATE(typename InputT, typename ReductionOp)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputT>)
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(const InputT& input, ReductionOp reduction_op)
  {
    if constexpr (is_commutative_associative_integral_reduction_v<T, ReductionOp>)
    {
      return integral_all_reduce_items(input, reduction_op);
    }
    else
    {
      return broadcast_from_lane0(WarpReduceT(temp_storage_).Reduce(input, reduction_op));
    }
  }

  //! @rst
  //! Computes a warp-wide reduction of up to ``valid_items`` items and returns the aggregate to every participating
  //! logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @smemwarpreuse
  //!
  //! Partial reductions use ``WarpReduce`` temporary storage; the integral fast path is not used.
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] input
  //!   Calling lane's input item.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator.
  //!
  //! @param[in] valid_items
  //!   Number of valid items across the logical warp. Must be in the range ``[1, LogicalWarpThreads]``.
  //!
  //! @return
  //!   The reduction aggregate of the valid items.
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int valid_items)
  {
    return broadcast_from_lane0(WarpReduceT(temp_storage_).Reduce(input, reduction_op, valid_items));
  }

  //! @}
};

CUB_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! @rst
//! The ``cub::WarpReduceBatchedBroadcast`` class provides :ref:`collective <collective-primitives>` methods for
//! computing batched warp-wide reductions whose aggregates are returned to every participating logical lane.
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
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``WarpReduceBatchedBroadcast`` class provides :ref:`collective <collective-primitives>` methods for computing
//! batched warp-wide reductions whose aggregates are returned to every participating logical lane.
//!
//! Overview
//! ++++++++
//!
//! - Supports "logical" warps smaller than the physical warp size (e.g., logical warps of 8 threads).
//! - Computes one independent reduction per batch.
//! - Returns every batch aggregate to every participating logical lane.
//! - The reduction operator must be commutative and associative. Use ``cub::WarpReduceBatched`` when aggregates only
//!   need to be returned to the owner lanes.
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - Uses warp shuffle instructions only, with no explicit synchronization between butterfly stages and no shared
//!   memory storage.
//! - Avoids sequentially invoking one warp-wide all-reduce per batch.
//! - For smaller than physical warp size logical warps, using ``SyncPhysicalWarp = true`` should in general give better
//!   performance than ``SyncPhysicalWarp = false``.
//!   Note that it can cause a deadlock if not all non-exited logical warps from the same physical warp participate in
//!   the reduction (due to branches).
//!   It also requires complete physical warps at the call site, and is unsafe for tail warps.
//!
//! Simple Example
//! ++++++++++++++
//!
//! @warpcollective{WarpReduceBatchedBroadcast}
//!
//! The code snippet below illustrates three batched sums across two 4-thread logical warps:
//!
//! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_broadcast_api.cu
//!     :language: c++
//!     :dedent:
//!     :start-after: example-begin warp-reduce-batched-broadcast-overview
//!     :end-before: example-end warp-reduce-batched-broadcast-overview
//!
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type.
//!
//! @tparam Batches
//!   The number of independent batches to reduce. Also corresponds to the size of the input and output ranges for each
//!   thread.
//!
//! @tparam LogicalWarpThreads
//!   **[optional]** The number of threads per "logical" warp (may be less than the number of hardware warp threads but
//!   has to be a power of two). Default is the warp size of the targeted CUDA compute-capability (e.g., 32 threads for
//!   SM80).
//!
//! @tparam SyncPhysicalWarp
//!   **[optional]** When true, synchronize the physical warp instead of only the logical warp. This requires complete
//!   physical warps at the call site, and all non-exited logical warps in each physical warp must participate.
template <typename T, int Batches, int LogicalWarpThreads = detail::warp_threads, bool SyncPhysicalWarp = false>
class WarpReduceBatchedBroadcast
{
  static_assert(Batches >= 0, "Batches must be >= 0");
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");

  template <typename InputT, typename OutputT, typename ReductionOp>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void check_constraints()
  {
    static_assert(cub::detail::is_fixed_size_random_access_range_v<InputT>,
                  "InputT must support operator[] and have a compile-time size");
    static_assert(cub::detail::is_fixed_size_random_access_range_v<OutputT>,
                  "OutputT must support operator[] and have a compile-time size");
    static_assert(cub::detail::static_size_v<InputT> == Batches, "Input size must match Batches");
    static_assert(cub::detail::static_size_v<OutputT> == Batches, "Output size must match Batches");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputT>, T>, "Input element type must match T");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<OutputT>, T>, "Output element type must match T");
    static_assert(cub::detail::has_binary_call_operator<ReductionOp, T>::value,
                  "ReductionOp must have the binary call operator: operator(T, T)");
  }

  template <typename OutputT, typename ReductionOp>
  static _CCCL_DEVICE _CCCL_FORCEINLINE void commutative_all_reduce(OutputT& outputs, ReductionOp reduction_op)
  {
    if constexpr (Batches == 0)
    {
      static_cast<void>(outputs);
      static_cast<void>(reduction_op);
      return;
    }

    const auto lane_id       = cub::detail::logical_lane_id<LogicalWarpThreads>();
    unsigned int member_mask = 0xFFFFFFFFu;
    if constexpr (!SyncPhysicalWarp)
    {
      const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
      member_mask                = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
    }

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
  //! \smemstorage{WarpReduceBatchedBroadcast}
  //!
  //! WarpReduceBatchedBroadcast uses only warp shuffle instructions, so this storage is an empty marker preserved for
  //! CUB collective constructor symmetry.
  struct TempStorage
  {};

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Logical warp and lane identifiers are constructed from ``threadIdx.x``.
  //! The storage object is an empty marker and does not consume shared memory.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpReduceBatchedBroadcast(TempStorage& temp_storage)
  {
    static_cast<void>(temp_storage);
  }

  //! @}
  //! @name Commutative reductions
  //! @{

  //! @rst
  //! Computes one warp-wide reduction per batch and returns every aggregate to every participating logical lane.
  //! The reduction operator must be commutative and associative.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @tparam OutputT
  //!   **[inferred]** Fixed-size random access range type receiving the batch aggregates for this lane.
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Commutative binary reduction operator type having member
  //!   ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items.
  //!
  //! @param[out] outputs
  //!   Statically-sized output range receiving ``Batches`` aggregate items.
  //!
  //! @param[in] reduction_op
  //!   Commutative binary reduction operator.
  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  CommutativeReduce(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op) const
  {
    check_constraints<InputT, OutputT, ReductionOp>();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      outputs[batch] = inputs[batch];
    }

    commutative_all_reduce(outputs, reduction_op);
  }

  //! @rst
  //! Computes one warp-wide reduction per batch and returns every aggregate to every participating logical lane.
  //! The reduction operator must be commutative and associative.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Commutative binary reduction operator type having member
  //!   ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items.
  //!
  //! @param[in] reduction_op
  //!   Commutative binary reduction operator.
  //!
  //! @return
  //!   The ``Batches`` warp-wide reduction aggregates.
  template <typename InputT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::array<T, Batches>
  CommutativeReduce(const InputT& inputs, ReductionOp reduction_op) const
  {
    ::cuda::std::array<T, Batches> outputs;
    CommutativeReduce(inputs, outputs, reduction_op);
    return outputs;
  }

  //! @}
  //! @name Sum reductions
  //! @{

  //! @rst
  //! Computes one warp-wide sum per batch and returns every aggregate to every participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @tparam OutputT
  //!   **[inferred]** Fixed-size random access range type receiving the batch aggregates for this lane.
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items.
  //!
  //! @param[out] outputs
  //!   Statically-sized output range receiving ``Batches`` aggregate items.
  template <typename InputT, typename OutputT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Sum(const InputT& inputs, OutputT& outputs) const
  {
    CommutativeReduce(inputs, outputs, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes one warp-wide sum per batch and returns every aggregate to every participating logical lane.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.5.
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Fixed-size random access range type containing this lane's input items.
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items.
  //!
  //! @return
  //!   The ``Batches`` warp-wide sum aggregates.
  template <typename InputT>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::array<T, Batches> Sum(const InputT& inputs) const
  {
    return CommutativeReduce(inputs, ::cuda::std::plus<>{});
  }

  //! @}
};

CUB_NAMESPACE_END

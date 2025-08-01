// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @rst
//! @file
//! cub::BlockReduceWarpReductions provides variants of warp-reduction-based parallel reduction
//! across a CUDA thread block. Supports non-commutative reduction operators.
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

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/__algorithm_>
#include <cuda/std/cmath>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! @rst
//! BlockReduceWarpReductions provides variants of warp-reduction-based parallel reduction
//! across a CUDA thread block. Supports non-commutative reduction operators.
//! @endrst
//!
//! @tparam T
//!   Data type being reduced
//!
//! @tparam BlockDimX
//!   The thread block length in threads along the X dimension
//!
//! @tparam BlockDimY
//!   The thread block length in threads along the Y dimension
//!
//! @tparam BlockDimZ
//!   The thread block length in threads along the Z dimension
//!
//! @tparam IsDeterministic
//!   Whether the reduction is deterministic
template <typename T, int BlockDimX, int BlockDimY, int BlockDimZ, bool IsDeterministic = true>
struct BlockReduceWarpReductions
{
  /// The thread block size in threads
  static constexpr int block_threads = BlockDimX * BlockDimY * BlockDimZ;

  /// Number of active warps
  static constexpr int warps = ::cuda::ceil_div(block_threads, warp_threads);

  /// The logical warp size for warp reductions
  static constexpr int logical_warp_size = _CUDA_VSTD::min(block_threads, warp_threads);

  /// Whether or not the logical warp size evenly divides the thread block size
  static constexpr bool even_warp_multiple = (block_threads % logical_warp_size == 0);

  using WarpReduceInternal = typename WarpReduce<T, logical_warp_size>::InternalWarpReduce;

  /// Shared memory storage layout type
  struct _TempStorage
  {
    /// Buffer for warp-synchronous reduction
    typename WarpReduceInternal::TempStorage warp_reduce[warps];

    /// Shared totals from each warp-synchronous reduction
    T warp_aggregates[warps];

    /// Shared prefix for the entire thread block
    T block_prefix;
  };

  using TempStorage = Uninitialized<_TempStorage>;

  // Thread fields
  _TempStorage& temp_storage;
  int linear_tid;
  int warp_id;
  int lane_id;

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceWarpReductions(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
      , warp_id((warps == 1) ? 0 : linear_tid / warp_threads)
      , lane_id(::cuda::ptx::get_sreg_laneid())
  {}

  //! @rst
  //! Returns block-wide aggregate in *thread*\ :sub:`0`.
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  //!
  //! @param[in] warp_aggregate
  //!   **[**\ *lane*\ :sub:`0` **only]** Warp-wide aggregate reduction of input items
  //!
  //! @param[in] num_valid
  //!   Number of valid elements (may be less than block_threads)
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(ReductionOp reduction_op, T warp_aggregate, int num_valid)
  {
    if (linear_tid == 0)
    {
      detail::uninitialized_copy_single(temp_storage.warp_aggregates, warp_aggregate);
    }

    __syncthreads();

    // Warp 0 already contributed its aggregate above since its also linear_tid == 0
    if (lane_id == 0 && warp_id != 0)
    {
      // TODO: replace this with other atomic operations when specified
      NV_IF_TARGET(NV_PROVIDES_SM_60,
                   (::cuda::atomic_ref<T, ::cuda::thread_scope_block> atomic_target(temp_storage.warp_aggregates[0]);
                    atomic_target.fetch_add(warp_aggregate, ::cuda::memory_order_relaxed);),
                   (atomicAdd(&temp_storage.warp_aggregates[0], warp_aggregate);));
    }

    __syncthreads();
    return temp_storage.warp_aggregates[0];
  }

  //! @rst
  //! Recursively applies warp aggregates using template unrolling for deterministic reduction.
  //! @endrst
  //!
  //! @tparam FullTile
  //!   **[inferred]** Whether this is a full tile
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type
  //!
  //! @tparam SuccessorWarp
  //!   **[inferred]** Index of the next warp to process
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  //!
  //! @param[in] warp_aggregate
  //!   **[**\ *lane*\ :sub:`0` **only]** Warp-wide aggregate reduction of input items
  //!
  //! @param[in] num_valid
  //!   Number of valid elements (may be less than block_threads)
  //!
  //! @param[in] successor_warp
  //!   Compile-time constant indicating the next warp index
  template <bool FullTile, typename ReductionOp, int SuccessorWarp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(
    ReductionOp reduction_op, T warp_aggregate, int num_valid, constant_t<SuccessorWarp> /*successor_warp*/)
  {
    if (FullTile || (SuccessorWarp * logical_warp_size < num_valid))
    {
      T addend       = temp_storage.warp_aggregates[SuccessorWarp];
      warp_aggregate = reduction_op(warp_aggregate, addend);
    }
    return ApplyWarpAggregates<FullTile>(reduction_op, warp_aggregate, num_valid, constant_v<SuccessorWarp + 1>);
  }

  //! @rst
  //! Template recursion termination case for deterministic warp aggregate application.
  //! @endrst
  //!
  //! @tparam FullTile
  //!   **[inferred]** Whether this is a full tile
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator (unused in termination case)
  //!
  //! @param[in] warp_aggregate
  //!   **[**\ *lane*\ :sub:`0` **only]** Warp-wide aggregate reduction of input items
  //!
  //! @param[in] num_valid
  //!   Number of valid elements (unused in termination case)
  //!
  //! @param[in] successor_warp
  //!   Compile-time constant indicating termination when equals warps
  template <bool FullTile, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(
    ReductionOp /*reduction_op*/, T warp_aggregate, int /*num_valid*/, constant_t<int{warps}> /*successor_warp*/)
  {
    return warp_aggregate;
  }

  /**
   * @brief Returns block-wide aggregate in <em>thread</em><sub>0</sub>.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] warp_aggregate
   *   <b>[<em>lane</em><sub>0</sub> only]</b> Warp-wide aggregate reduction of input items
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than block_threads)
   */
  template <bool FullTile, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(ReductionOp reduction_op, T warp_aggregate, int num_valid)
  {
    // Share lane aggregates
    if (lane_id == 0)
    {
      detail::uninitialized_copy_single(temp_storage.warp_aggregates + warp_id, warp_aggregate);
    }

    __syncthreads();

    // Update total aggregate in warp 0, lane 0
    if (linear_tid == 0)
    {
      warp_aggregate = ApplyWarpAggregates<FullTile>(reduction_op, warp_aggregate, num_valid, constant_v<1>);
    }

    return warp_aggregate;
  }

  //! @rst
  //! Computes a thread block-wide reduction using addition (+/cuda::std::plus<>) as the reduction operator.
  //! The first num_valid threads each contribute one reduction partial. The return value is
  //! only valid for *thread*\ :sub:`0`.
  //! @endrst
  //!
  //! @tparam FullTile
  //!   **[inferred]** Whether this is a full tile
  //!
  //! @param[in] input
  //!   Calling thread's input partial reductions
  //!
  //! @param[in] num_valid
  //!   Number of valid elements (may be less than block_threads)
  template <bool FullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    ::cuda::std::plus<> reduction_op;
    const int warp_offset = (warp_id * logical_warp_size);
    const int warp_num_valid =
      ((FullTile && even_warp_multiple) || (warp_offset + logical_warp_size <= num_valid))
        ? logical_warp_size
        : num_valid - warp_offset;

    // Warp reduction in every warp
    T warp_aggregate = WarpReduceInternal(temp_storage.warp_reduce[warp_id])
                         .template Reduce<(FullTile && even_warp_multiple)>(input, warp_num_valid, reduction_op);

    // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
    if constexpr (IsDeterministic)
    {
      return ApplyWarpAggregates<FullTile>(reduction_op, warp_aggregate, num_valid);
    }
    else
    {
      return ApplyWarpAggregates(reduction_op, warp_aggregate, num_valid);
    }
  }

  //! @rst
  //! Computes a thread block-wide reduction using the specified reduction operator.
  //! The first num_valid threads each contribute one reduction partial.
  //! The return value is only valid for *thread*\ :sub:`0`.
  //! @endrst
  //!
  //! @tparam FullTile
  //!   **[inferred]** Whether this is a full tile
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type
  //!
  //! @param[in] input
  //!   Calling thread's input partial reductions
  //!
  //! @param[in] num_valid
  //!   Number of valid elements (may be less than block_threads)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  template <bool FullTile, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int num_valid, ReductionOp reduction_op)
  {
    const int warp_offset = warp_id * logical_warp_size;
    const int warp_num_valid =
      ((FullTile && even_warp_multiple) || (warp_offset + logical_warp_size <= num_valid))
        ? logical_warp_size
        : num_valid - warp_offset;

    // Warp reduction in every warp
    const T warp_aggregate = WarpReduceInternal(temp_storage.warp_reduce[warp_id])
                               .template Reduce<(FullTile && even_warp_multiple)>(input, warp_num_valid, reduction_op);

    // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
    if constexpr (IsDeterministic)
    {
      return ApplyWarpAggregates<FullTile>(reduction_op, warp_aggregate, num_valid);
    }
    else
    {
      return ApplyWarpAggregates(reduction_op, warp_aggregate, num_valid);
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END

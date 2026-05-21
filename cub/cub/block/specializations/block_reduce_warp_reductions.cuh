// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__algorithm/min.h>

#include <nv/target>

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
  static constexpr int threads_per_block = BlockDimX * BlockDimY * BlockDimZ;

  /// Number of active warps
  static constexpr int warps = ::cuda::ceil_div(threads_per_block, warp_threads);

  /// The logical warp size for warp reductions
  static constexpr int logical_warp_size = ::cuda::std::min(threads_per_block, warp_threads);

  /// Whether or not the logical warp size evenly divides the thread block size
  static constexpr bool even_warp_multiple = (threads_per_block % logical_warp_size == 0);

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
  //! Cooperatively reduces warp aggregates using warp 0.

  //! @endrst
  //!
  //! @tparam FullTile
  //!   **[inferred]** Whether this is a full tile
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type
  template <bool FullTile, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(ReductionOp reduction_op, T warp_aggregate, int num_valid)
  {
    // Share lane aggregates
    if (lane_id == 0)
    {
      detail::uninitialized_copy_single(temp_storage.warp_aggregates + warp_id, warp_aggregate);
    }

    __syncthreads();

    // Warp 0 cooperatively reduces all warp aggregates
    if (warp_id == 0)
    {
      const int num_warps = FullTile ? int(warps) : static_cast<int>(::cuda::ceil_div(num_valid, logical_warp_size));

      T val{};
      if (lane_id < num_warps)
      {
        val = temp_storage.warp_aggregates[lane_id];
      }

      // Fast path: redux intrinsic for eligible types/ops on SM80+ (full tile only)
      if constexpr (FullTile && is_redux_enabled_cuda_operator<ReductionOp, T>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_80, ({
                       constexpr unsigned mask = (warps == warp_threads) ? 0xFFFFFFFFu : ((1u << warps) - 1u);
                       if (lane_id < warps)
                       {
                         warp_aggregate = reduce_op_sync(val, mask, reduction_op);
                       }
                       return warp_aggregate;
                     }))
      }

      // Shuffle-based warp reduction fallback
      constexpr bool all_warps_valid = (FullTile && (warps == warp_threads));
      NullType dummy_storage;
      warp_aggregate =
        WarpReduceShfl<T, warp_threads>(dummy_storage).template Reduce<all_warps_valid>(val, num_warps, reduction_op);
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
  //!   Number of valid elements (may be less than threads_per_block)
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
    return ApplyWarpAggregates<FullTile>(reduction_op, warp_aggregate, num_valid);
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
  //!   Number of valid elements (may be less than threads_per_block)
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
    return ApplyWarpAggregates<FullTile>(reduction_op, warp_aggregate, num_valid);
  }
};
} // namespace detail

CUB_NAMESPACE_END

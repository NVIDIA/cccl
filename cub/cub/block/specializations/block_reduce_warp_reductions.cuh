/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/__algorithm/min.h>

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
  static constexpr int logical_warp_size = ::cuda::std::min(block_threads, warp_threads);

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
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregatesNonDeterministic(ReductionOp reduction_op, T warp_aggregate)
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
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int warp_idx = 1; warp_idx < warps; ++warp_idx)
      {
        if (FullTile || (warp_idx * logical_warp_size < num_valid))
        {
          T addend       = temp_storage.warp_aggregates[warp_idx];
          warp_aggregate = reduction_op(warp_aggregate, addend);
        }
      }
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
      return ApplyWarpAggregatesNonDeterministic(reduction_op, warp_aggregate);
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
      return ApplyWarpAggregatesNonDeterministic(reduction_op, warp_aggregate);
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END

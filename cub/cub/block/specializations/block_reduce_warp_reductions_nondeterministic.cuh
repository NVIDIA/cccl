// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 * cub::BlockReduceWarpReductionsNondeterministic provides variants of warp-reduction-based parallel reduction
 * across a CUDA thread block. Supports non-commutative reduction operators.
 */

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

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief BlockReduceWarpReductionsNondeterministic provides variants of warp-reduction-based parallel reduction
 *        across a CUDA thread block. Supports non-commutative reduction operators.
 * @tparam T
 *   Data type being reduced
 *
 * @tparam BLOCK_DIM_X
 *   The thread block length in threads along the X dimension
 *
 * @tparam BLOCK_DIM_Y
 *   The thread block length in threads along the Y dimension
 *
 * @tparam BLOCK_DIM_Z
 *   The thread block length in threads along the Z dimension
 */
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
struct BlockReduceWarpReductionsNondeterministic
{
  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  /// Number of warp threads
  static constexpr int WARP_THREADS = warp_threads;

  /// Number of active warps
  static constexpr int WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;

  /// The logical warp size for warp reductions
  static constexpr int LOGICAL_WARP_SIZE = ::cuda::std::min(BLOCK_THREADS, WARP_THREADS);

  /// Whether or not the logical warp size evenly divides the thread block size
  static constexpr bool EVEN_WARP_MULTIPLE = (BLOCK_THREADS % LOGICAL_WARP_SIZE == 0);

  ///  WarpReduce utility type
  using WarpReduceInternal = typename WarpReduce<T, LOGICAL_WARP_SIZE>::InternalWarpReduce;

  /// Shared memory storage layout type
  struct _TempStorage
  {
    /// Buffer for warp-synchronous reduction
    typename WarpReduceInternal::TempStorage warp_reduce[WARPS];

    /// Shared totals from each warp-synchronous reduction
    T warp_aggregates[WARPS];

    /// Shared prefix for the entire thread block
    T block_prefix;
  };

  using TempStorage = Uninitialized<_TempStorage>;

  // Thread fields
  _TempStorage& temp_storage;
  int linear_tid;
  int warp_id;
  int lane_id;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceWarpReductionsNondeterministic(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
      , warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS)
      , lane_id(::cuda::ptx::get_sreg_laneid())
  {}

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
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T ApplyWarpAggregates(ReductionOp reduction_op, T warp_aggregate, int num_valid)
  {
    if (linear_tid == 0)
    {
      detail::uninitialized_copy_single(temp_storage.warp_aggregates, warp_aggregate);
    }

    __syncthreads();

    if (lane_id == 0 && warp_id != 0)
    {
      // TODO: replace this with other atomic operations when specified
      ::cuda::atomic_ref<T> atomic_target(temp_storage.warp_aggregates[0]);
      atomic_target += warp_aggregate;
    }

    __syncthreads();
    return temp_storage.warp_aggregates[0];
  }

  /**
   * @brief Computes a thread block-wide reduction using addition (+) as the reduction operator.
   *        The first num_valid threads each contribute one reduction partial. The return value is
   *        only valid for thread<sub>0</sub>.
   *
   * @param[in] input
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   */
  template <bool FullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    ::cuda::std::plus<> reduction_op;
    const int warp_offset = (warp_id * LOGICAL_WARP_SIZE);
    const int warp_num_valid =
      ((FullTile && EVEN_WARP_MULTIPLE) || (warp_offset + LOGICAL_WARP_SIZE <= num_valid))
        ? LOGICAL_WARP_SIZE
        : num_valid - warp_offset;

    // Warp reduction in every warp
    T warp_aggregate = WarpReduceInternal(temp_storage.warp_reduce[warp_id])
                         .template Reduce<(FullTile && EVEN_WARP_MULTIPLE)>(input, warp_num_valid, reduction_op);

    // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
    return ApplyWarpAggregates(reduction_op, warp_aggregate, num_valid);
  }

  /**
   * @brief Computes a thread block-wide reduction using the specified reduction operator.
   *        The first num_valid threads each contribute one reduction partial.
   *        The return value is only valid for thread<sub>0</sub>.
   *
   * @param[in] input
   *   Calling thread's input partial reductions
   *
   * @param[in] num_valid
   *   Number of valid elements (may be less than BLOCK_THREADS)
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool FullTile, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int num_valid, ReductionOp reduction_op)
  {
    const int warp_offset = warp_id * LOGICAL_WARP_SIZE;
    const int warp_num_valid =
      ((FullTile && EVEN_WARP_MULTIPLE) || (warp_offset + LOGICAL_WARP_SIZE <= num_valid))
        ? LOGICAL_WARP_SIZE
        : num_valid - warp_offset;

    // Warp reduction in every warp
    const T warp_aggregate = WarpReduceInternal(temp_storage.warp_reduce[warp_id])
                               .template Reduce<(FullTile && EVEN_WARP_MULTIPLE)>(input, warp_num_valid, reduction_op);

    // Update outputs and block_aggregate with warp-wide aggregates from lane-0s
    return ApplyWarpAggregates(reduction_op, warp_aggregate, num_valid);
  }
};
} // namespace detail

CUB_NAMESPACE_END

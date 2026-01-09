// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned across a CUDA thread
 * warp.
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

#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T reduce_op_sync(T input, const uint32_t mask, ReductionOp)
{
  static_assert(::cuda::std::is_integral_v<T>, "T must be an integral type");
  static_assert(sizeof(T) <= sizeof(unsigned), "T must be less than or equal to unsigned");
  using promoted_t = ::cuda::std::conditional_t<::cuda::std::is_unsigned_v<T>, unsigned, int>;
  if constexpr (is_cuda_maximum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_max_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_min_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_add_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_and_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_and_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_or_sync(mask, static_cast<promoted_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_xor_sync(mask, static_cast<promoted_t>(input)));
  }
  else
  {
    _CCCL_UNREACHABLE();
    return T{};
  }
}

/**
 * @brief WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being reduced
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp (must be a power-of-two)
 */
template <typename T, int LOGICAL_WARP_THREADS>
struct WarpReduceShfl
{
  static_assert(::cuda::is_power_of_two(LOGICAL_WARP_THREADS), "LOGICAL_WARP_THREADS must be a power of two");

  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == warp_threads);

  /// The number of warp reduction steps
  static constexpr int STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE;

  /// Number of logical warps in a PTX warp
  static constexpr int LOGICAL_WARPS = warp_threads / LOGICAL_WARP_THREADS;

  /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
  static constexpr unsigned SHFL_C = (warp_threads - LOGICAL_WARP_THREADS) << 8;

  /// Shared memory storage layout type
  using TempStorage = NullType;

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  /// Lane index in logical warp
  int lane_id;

  /// Logical warp index in 32-thread physical warp
  int warp_id;

  /// 32-thread physical warp member mask of logical warp
  ::cuda::std::uint32_t member_mask;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceShfl(TempStorage& /*temp_storage*/)
      : lane_id(static_cast<int>(::cuda::ptx::get_sreg_laneid()))
      , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {
    if (!IS_ARCH_WARP)
    {
      lane_id = lane_id % LOGICAL_WARP_THREADS;
    }
  }

  //---------------------------------------------------------------------
  // Reduction steps
  //---------------------------------------------------------------------

  /**
   * @brief Reduction (specialized for summation across uint32 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int
  ReduceStep(unsigned int input, ::cuda::std::plus<> /*reduction_op*/, int last_lane, int offset)
  {
    unsigned int output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.u32 r0, r0, %4;"
      "  mov.u32 %0, r0;"
      "}"
      : "=r"(output)
      : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across fp32 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE float
  ReduceStep(float input, ::cuda::std::plus<> /*reduction_op*/, int last_lane, int offset)
  {
    float output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .f32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.f32 r0, r0, %4;"
      "  mov.f32 %0, r0;"
      "}"
      : "=f"(output)
      : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across unsigned long long types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned long long
  ReduceStep(unsigned long long input, ::cuda::std::plus<> /*reduction_op*/, int last_lane, int offset)
  {
    unsigned long long output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 %0, {lo, hi};"
      "  @p add.u64 %0, %0, %1;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across long long types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE long long
  ReduceStep(long long input, ::cuda::std::plus<> /*reduction_op*/, int last_lane, int offset)
  {
    long long output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 %0, {lo, hi};"
      "  @p add.s64 %0, %0, %1;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for summation across double types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE double
  ReduceStep(double input, ::cuda::std::plus<> /*reduction_op*/, int last_lane, int offset)
  {
    double output;
    int shfl_c = last_lane | SHFL_C; // Shuffle control (mask and last_lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  .reg .f64 r0;"
      "  mov.b64 %0, %1;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.f64 %0, %0, r0;"
      "}"
      : "=d"(output)
      : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /**
   * @brief Reduction (specialized for swizzled ReduceByKeyOp<::cuda::std::plus<>> across
   *        KeyValuePair<KeyT, ValueT> types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename ValueT, typename KeyT>
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePair<KeyT, ValueT> ReduceStep(
    KeyValuePair<KeyT, ValueT> input,
    SwizzleScanOp<ReduceByKeyOp<::cuda::std::plus<>>> /*reduction_op*/,
    int last_lane,
    int offset)
  {
    KeyValuePair<KeyT, ValueT> output;

    KeyT other_key = ShuffleDown<LOGICAL_WARP_THREADS>(input.key, offset, last_lane, member_mask);

    output.key   = input.key;
    output.value = ReduceStep(input.value, ::cuda::std::plus<>{}, last_lane, offset);

    if (input.key != other_key)
    {
      output.value = input.value;
    }

    return output;
  }

  /**
   * @brief Reduction (specialized for swizzled ReduceBySegmentOp<cuda::std::plus<>> across
   *        KeyValuePair<OffsetT, ValueT> types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename ValueT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, ValueT> ReduceStep(
    KeyValuePair<OffsetT, ValueT> input,
    SwizzleScanOp<ReduceBySegmentOp<::cuda::std::plus<>>> /*reduction_op*/,
    int last_lane,
    int offset)
  {
    KeyValuePair<OffsetT, ValueT> output;

    output.value = ReduceStep(input.value, ::cuda::std::plus<>{}, last_lane, offset);
    output.key   = ReduceStep(input.key, ::cuda::std::plus<>{}, last_lane, offset);

    if (input.key > 0)
    {
      output.value = input.value;
    }

    return output;
  }

  /**
   * @brief Reduction step (generic)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename _Tp, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE _Tp ReduceStep(_Tp input, ReductionOp reduction_op, int last_lane, int offset)
  {
    _Tp output = input;

    _Tp temp = ShuffleDown<LOGICAL_WARP_THREADS>(output, offset, last_lane, member_mask);

    // Perform reduction op if valid
    if (offset + lane_id <= last_lane)
    {
      output = reduction_op(input, temp);
    }

    return output;
  }

  //---------------------------------------------------------------------
  // Templated reduction iteration
  //---------------------------------------------------------------------

  /**
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   */
  template <typename ReductionOp, int STEP>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ReduceStep(T& input, ReductionOp reduction_op, int last_lane, constant_t<STEP> /*step*/)
  {
    input = ReduceStep(input, reduction_op, last_lane, 1 << STEP);
    ReduceStep(input, reduction_op, last_lane, constant_v<STEP + 1>);
  }

  /**
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   */
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ReduceStep(T& /*input*/, ReductionOp /*reduction_op*/, int /*last_lane*/, constant_t<STEPS> /*step*/)
  {}

  //---------------------------------------------------------------------
  // Reduction operations
  //---------------------------------------------------------------------

  /**
   * @brief Reduction
   *
   * @tparam ALL_LANES_VALID
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int valid_items, ReductionOp reduction_op)
  {
    // Dispatch to more efficient intrinsics when applicable
    if constexpr (ALL_LANES_VALID && ::cuda::std::is_integral_v<T> && sizeof(T) <= sizeof(unsigned)
                  && (is_cuda_minimum_maximum_v<ReductionOp, T> || is_cuda_std_plus_v<ReductionOp, T>
                      || is_cuda_std_bitwise_v<ReductionOp, T>) )
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80, (return reduce_op_sync(input, member_mask, reduction_op);))
    }
    T output = input;
    // Template-iterate reduction steps
    const int last_lane = (ALL_LANES_VALID) ? LOGICAL_WARP_THREADS - 1 : valid_items - 1;
    ReduceStep(output, reduction_op, last_lane, constant_v<0>);
    return output;
  }

  /**
   * @brief Segmented reduction
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op)
  {
    // Get the start flags for each thread in the warp.
    unsigned warp_flags = __ballot_sync(member_mask, flag);

    // Convert to tail-segmented
    if (HEAD_SEGMENTED)
    {
      warp_flags >>= 1;
    }

    // Mask out the bits below the current thread
    warp_flags &= ::cuda::ptx::get_sreg_lanemask_ge();

    // Mask of physical lanes outside the logical warp and convert to logical lanemask
    if (!IS_ARCH_WARP)
    {
      warp_flags = (warp_flags & member_mask) >> (warp_id * LOGICAL_WARP_THREADS);
    }

    // Mask in the last lane of logical warp
    warp_flags |= 1u << (LOGICAL_WARP_THREADS - 1);

    // Find the next set flag
    int last_lane = ::cuda::std::countr_zero(warp_flags);

    T output = input;
    // Template-iterate reduction steps
    ReduceStep(output, reduction_op, last_lane, constant_v<0>);

    return output;
  }
};
} // namespace detail

CUB_NAMESPACE_END

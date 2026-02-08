// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned
 * across a CUDA thread warp.
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

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being scanned
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp (must be a power-of-two)
 */
template <typename T, int LOGICAL_WARP_THREADS>
struct WarpScanShfl
{
  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == warp_threads);

  /// The number of warp scan steps
  static constexpr int STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE;

  /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
  static constexpr int SHFL_C = (warp_threads - LOGICAL_WARP_THREADS) << 8;

  template <typename S>
  struct IntegerTraits
  {
    /// Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per
    /// exchange
    static constexpr bool IS_SMALL_UNSIGNED =
      ::cuda::std::is_integral_v<S> && ::cuda::std::is_unsigned_v<S> && (sizeof(S) <= sizeof(unsigned int));
  };

  /// Shared memory storage layout type
  struct TempStorage
  {};

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  /// Lane index in logical warp
  unsigned int lane_id;

  /// Logical warp index in 32-thread physical warp
  unsigned int warp_id;

  /// 32-thread physical warp member mask of logical warp
  unsigned int member_mask;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpScanShfl(TempStorage& /*temp_storage*/)
      : lane_id(::cuda::ptx::get_sreg_laneid())
      , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {
    if (!IS_ARCH_WARP)
    {
      lane_id = lane_id % LOGICAL_WARP_THREADS;
    }
  }

  //---------------------------------------------------------------------
  // Inclusive scan steps
  //---------------------------------------------------------------------

  /**
   * @brief Inclusive prefix scan step (specialized for summation across int32 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE int
  InclusiveScanStep(int input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    int output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .s32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.s32 r0, r0, %4;"
      "  mov.s32 %0, r0;"
      "}"
      : "=r"(output)
      : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across uint32 types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int
  InclusiveScanStep(unsigned int input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    unsigned int output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.u32 r0, r0, %4;"
      "  mov.u32 %0, r0;"
      "}"
      : "=r"(output)
      : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across fp32 types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE float
  InclusiveScanStep(float input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    float output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .f32 r0;"
      "  .reg .pred p;"
      "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "  @p add.f32 r0, r0, %4;"
      "  mov.f32 %0, r0;"
      "}"
      : "=f"(output)
      : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across unsigned long long types)
   *
   * @param[in]  input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned long long
  InclusiveScanStep(unsigned long long input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    unsigned long long output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u64 r0;"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
      "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.u64 r0, r0, %4;"
      "  mov.u64 %0, r0;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across long long types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE long long
  InclusiveScanStep(long long input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    long long output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .s64 r0;"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
      "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.s64 r0, r0, %4;"
      "  mov.s64 %0, r0;"
      "}"
      : "=l"(output)
      : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));

    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for summation across fp64 types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE double
  InclusiveScanStep(double input, ::cuda::std::plus<> /*scan_op*/, int first_lane, int offset)
  {
    double output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
      "{"
      "  .reg .u32 lo;"
      "  .reg .u32 hi;"
      "  .reg .pred p;"
      "  .reg .f64 r0;"
      "  mov.b64 %0, %1;"
      "  mov.b64 {lo, hi}, %1;"
      "  shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
      "  shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
      "  mov.b64 r0, {lo, hi};"
      "  @p add.f64 %0, %0, r0;"
      "}"
      : "=d"(output)
      : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));

    return output;
  }

  /*
  /// Inclusive prefix scan (specialized for ReduceBySegmentOp<::cuda::std::plus<>> across KeyValuePair<OffsetT, Value>
  /// types)
  template <typename Value, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, Value> InclusiveScanStep(
    KeyValuePair<OffsetT, Value> input, ///< [in] Calling thread's input item.
    ReduceBySegmentOp<::cuda::std::plus<>> scan_op, ///< [in] Binary scan operator
    int first_lane, ///< [in] Index of first lane in segment
    int offset) ///< [in] Up-offset to pull from
  {
    KeyValuePair<OffsetT, Value> output;
    output.value = InclusiveScanStep(
      input.value, ::cuda::std::plus<>{}, first_lane, offset, Int2Type<IntegerTraits<Value>::IS_SMALL_UNSIGNED>());
    output.key = InclusiveScanStep(
      input.key, ::cuda::std::plus<>{}, first_lane, offset, Int2Type<IntegerTraits<OffsetT>::IS_SMALL_UNSIGNED>());

    if (input.key > 0)
      output.value = input.value;

    return output;
  }
  */

  /**
   * @brief Inclusive prefix scan step (generic)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename _Tp, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _Tp InclusiveScanStep(_Tp input, ScanOpT scan_op, int first_lane, int offset)
  {
    _Tp temp = ShuffleUp<LOGICAL_WARP_THREADS>(input, offset, first_lane, member_mask);

    // Perform scan op if from a valid peer
    _Tp output = scan_op(temp, input);
    if (static_cast<int>(lane_id) < first_lane + offset)
    {
      output = input;
    }

    return output;
  }

  /**
   * @brief Partial inclusive prefix scan step (generic)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename _Tp, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _Tp
  InclusiveScanStepPartial(_Tp input, ScanOpT scan_op, int valid_items, int first_lane, int offset)
  {
    _CCCL_ASSERT((first_lane >= 0) && (first_lane <= static_cast<int>(lane_id)),
                 "first_lane must be in range [0, lane_id]");
    _CCCL_ASSERT((offset > 0) && (offset < LOGICAL_WARP_THREADS),
                 "offset must be in the range [1, LOGICAL_WARP_THREADS)");
    _CCCL_ASSERT(::cuda::std::has_single_bit(static_cast<unsigned>(offset)), "offset must be a power of two");
    _Tp temp = ::cuda::device::warp_shuffle_up<LOGICAL_WARP_THREADS>(input, offset, member_mask);

    // Perform scan op if from a valid peer
    _Tp output = input;
    if (static_cast<int>(lane_id) >= first_lane + offset && static_cast<int>(lane_id) < valid_items)
    {
      output = scan_op(temp, input);
    }
    return output;
  }

  /**
   * @brief Inclusive prefix scan step (specialized for small integers size 32b or less)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   *
   * @param[in] is_small_unsigned
   *   Marker type indicating whether T is a small integer
   */
  template <typename _Tp, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _Tp InclusiveScanStep(
    _Tp input, ScanOpT scan_op, int first_lane, int offset, ::cuda::std::true_type /*is_small_unsigned*/)
  {
    return InclusiveScanStep(input, scan_op, first_lane, offset);
  }

  /**
   * @brief Inclusive prefix scan step (specialized for types other than small integers size
   *        32b or less)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] first_lane
   *   Index of first lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   *
   * @param[in] is_small_unsigned
   *   Marker type indicating whether T is a small integer
   */
  template <typename _Tp, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE _Tp InclusiveScanStep(
    _Tp input, ScanOpT scan_op, int first_lane, int offset, ::cuda::std::false_type /*is_small_unsigned*/)
  {
    return InclusiveScanStep(input, scan_op, first_lane, offset);
  }

  /******************************************************************************
   * Interface
   ******************************************************************************/

  //---------------------------------------------------------------------
  // Broadcast
  //---------------------------------------------------------------------

  /**
   * @brief Broadcast
   *
   * @param[in] input
   *   The value to broadcast
   *
   * @param[in] src_lane
   *   Which warp lane is to do the broadcasting
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE T Broadcast(T input, int src_lane)
  {
    return ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
  }

  //---------------------------------------------------------------------
  // Inclusive operations
  //---------------------------------------------------------------------

  /**
   * @brief Inclusive scan
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename _Tp, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(_Tp input, _Tp& inclusive_output, ScanOpT scan_op)
  {
    inclusive_output = input;

    // Iterate scan steps
    int segment_first_lane = 0;

    // Iterate scan steps
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      inclusive_output = InclusiveScanStep(
        inclusive_output,
        scan_op,
        segment_first_lane,
        (1 << STEP),
        bool_constant_v<IntegerTraits<T>::IS_SMALL_UNSIGNED>);
    }
  }

  /**
   * @brief Inclusive scan (partial)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   */
  template <typename _Tp, typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScanPartial(_Tp input, _Tp& inclusive_output, ScanOpT scan_op, int valid_items)
  {
    if (static_cast<int>(lane_id) < valid_items)
    {
      inclusive_output = input;
    }
    // Iterate scan steps
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int step = 0; step < STEPS; step++)
    {
      constexpr int segment_first_lane = 0;
      inclusive_output =
        InclusiveScanStepPartial(inclusive_output, scan_op, valid_items, segment_first_lane, (1 << step));
    }
  }

  /**
   * @brief Inclusive scan, specialized for reduce-value-by-key
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename KeyT, typename ValueT, typename ReductionOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(
    KeyValuePair<KeyT, ValueT> input, KeyValuePair<KeyT, ValueT>& inclusive_output, ReduceByKeyOp<ReductionOpT> scan_op)
  {
    inclusive_output = input;

    KeyT pred_key = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive_output.key, 1, 0, member_mask);

    unsigned int ballot = __ballot_sync(member_mask, (pred_key != inclusive_output.key));

    // Mask away all lanes greater than ours
    ballot = ballot & ::cuda::ptx::get_sreg_lanemask_le();

    // Find index of first set bit
    int segment_first_lane = ::cuda::std::__bit_log2(ballot);

    // Iterate scan steps
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      inclusive_output.value = InclusiveScanStep(
        inclusive_output.value,
        scan_op.op,
        segment_first_lane,
        (1 << STEP),
        bool_constant_v<IntegerTraits<T>::IS_SMALL_UNSIGNED>);
    }
  }

  /**
   * @brief Inclusive scan with aggregate
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[out] warp_aggregate
   *   Warp-wide aggregate reduction of input items
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOpT scan_op, T& warp_aggregate)
  {
    InclusiveScan(input, inclusive_output, scan_op);

    // Grab aggregate from last warp lane
    warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive_output, LOGICAL_WARP_THREADS - 1, member_mask);
  }

  /**
   * @brief Partial inclusive scan with aggregate
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   *
   * @param[out] warp_aggregate
   *   Warp-wide aggregate reduction of input items
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScanPartial(T input, T& inclusive_output, ScanOpT scan_op, int valid_items, T& warp_aggregate)
  {
    InclusiveScanPartial(input, inclusive_output, scan_op, valid_items);

    // Grab aggregate from last valid warp lane
    const int last_valid_lane = ::cuda::std::clamp(valid_items - 1, 0, LOGICAL_WARP_THREADS - 1);
    warp_aggregate =
      ::cuda::device::warp_shuffle_idx<LOGICAL_WARP_THREADS>(inclusive_output, last_valid_lane, member_mask);
  }

  //---------------------------------------------------------------------
  // Get exclusive from inclusive
  //---------------------------------------------------------------------

  /**
   * @brief Update inclusive and exclusive using input and inclusive
   *
   * @param[in] input
   *
   * @param[out] inclusive
   *
   * @param[out] exclusive
   *
   * @param[in] scan_op
   *
   * @param[in] is_integer
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T /*input*/, T& inclusive, T& exclusive, ScanOpT /*scan_op*/, IsIntegerT /*is_integer*/)
  {
    // initial value unknown
    exclusive = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask);
  }

  /**
   * @brief Update inclusive and exclusive using input and inclusive (specialized for summation of
   *        integer types)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T input, T& inclusive, T& exclusive, ::cuda::std::plus<> /*scan_op*/, ::cuda::std::true_type /*is_integer*/)
  {
    // initial value presumed 0
    exclusive = inclusive - input;
  }

  /**
   * @brief Update inclusive and exclusive using initial value using input, inclusive, and initial
   *        value
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T /*input*/, T& inclusive, T& exclusive, ScanOpT scan_op, T initial_value, IsIntegerT /*is_integer*/)
  {
    inclusive = scan_op(initial_value, inclusive);
    exclusive = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask);

    if (lane_id == 0)
    {
      exclusive = initial_value;
    }
  }

  /**
   * @brief Update inclusive and exclusive using initial value using input and inclusive
   *        (specialized for summation of integer types)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T input,
    T& inclusive,
    T& exclusive,
    ::cuda::std::plus<> scan_op,
    T initial_value,
    ::cuda::std::true_type /*is_integer*/)
  {
    inclusive = scan_op(initial_value, inclusive);
    exclusive = inclusive - input;
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input and inclusive
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Update(T input, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT scan_op, IsIntegerT is_integer)
  {
    warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive, LOGICAL_WARP_THREADS - 1, member_mask);
    Update(input, inclusive, exclusive, scan_op, is_integer);
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input, inclusive, and initial
   *        value
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T input, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT scan_op, T initial_value, IsIntegerT is_integer)
  {
    warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive, LOGICAL_WARP_THREADS - 1, member_mask);
    Update(input, inclusive, exclusive, scan_op, initial_value, is_integer);
  }

  /**
   * @brief Compute valid elements of the exclusive scan using input and/or inclusive.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] inclusive
   *   Calling thread's item of the previously computed inclusive scan
   *
   * @param[out] exclusive
   *   Calling thread's item of the exclusive scan (undefined for lane 0)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  UpdatePartial([[maybe_unused]] T input, T& inclusive, T& exclusive, [[maybe_unused]] ScanOpT scan_op, int valid_items)
  {
    if constexpr (::cuda::std::is_integral_v<T> && cub::detail::is_cuda_std_plus_v<ScanOpT, T>)
    {
      // initial value presumed 0
      if (static_cast<int>(lane_id) < valid_items)
      {
        exclusive = inclusive - input;
      }
    }
    else
    {
      // initial value unknown
      T temp = ::cuda::device::warp_shuffle_up<LOGICAL_WARP_THREADS>(inclusive, 1, member_mask);
      if (static_cast<int>(lane_id) < valid_items)
      {
        exclusive = temp;
      }
    }
  }

  /**
   * @brief Update valid elements of the inclusive scan with the initial value and compute valid
   *        elements of the exclusive scan using input and/or the updated inclusive scan.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in, out] inclusive
   *   Calling thread's item of the previously computed inclusive scan to be updated with initial
   *   value
   *
   * @param[out] exclusive
   *   Calling thread's item of the exclusive scan (undefined for lane 0)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   *
   * @param[in] initial_value
   *   Initial value to seed the scan (uniform across warp)
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  UpdatePartial(T input, T& inclusive, T& exclusive, ScanOpT scan_op, int valid_items, T initial_value)
  {
    // Update inclusive
    if (static_cast<int>(lane_id) < valid_items)
    {
      inclusive = scan_op(initial_value, inclusive);
    }
    // Get exclusive
    UpdatePartial(input, inclusive, exclusive, scan_op, valid_items);

    if constexpr (!(::cuda::std::is_integral_v<T> && cub::detail::is_cuda_std_plus_v<ScanOpT, T>) )
    {
      // Correct first element of exclusive
      if ((lane_id == 0u) && (valid_items > 0))
      {
        exclusive = initial_value;
      }
    }
  }

  /**
   * @brief Compute valid elements of the exclusive scan and the warp aggregate using input and/or
   *        the inclusive scan.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] inclusive
   *   Calling thread's item of the previously computed inclusive scan
   *
   * @param[out] exclusive
   *   Calling thread's item of the exclusive scan (undefined for lane 0)
   *
   * @param[out] warp_aggregate
   *   Warp-wide aggregate reduction of valid input items.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  UpdatePartial(T input, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT scan_op, int valid_items)
  {
    // Get aggregate
    const int last_valid_lane = ::cuda::std::clamp(valid_items - 1, 0, LOGICAL_WARP_THREADS - 1);
    warp_aggregate = ::cuda::device::warp_shuffle_idx<LOGICAL_WARP_THREADS>(inclusive, last_valid_lane, member_mask);
    // Compute exclusive
    UpdatePartial(input, inclusive, exclusive, scan_op, valid_items);
  }

  /**
   * @brief Update valid elements of the inclusive scan with the initial value and compute valid
   *        elements of the exclusive scan and the aggregate using input and/or the updated
   *        inclusive scan.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in, out] inclusive
   *   Calling thread's item of the previously computed inclusive scan
   *
   * @param[out] exclusive
   *   Calling thread's item of the exclusive scan (undefined for lane 0)
   *
   * @param[out] warp_aggregate
   *   Warp-wide aggregate reduction of valid input items.
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items in warp
   *
   * @param[in] initial_value
   *   Initial value to seed the scan (uniform across warp)
   */
  template <typename ScanOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void UpdatePartial(
    T input, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT scan_op, int valid_items, T initial_value)
  {
    // Get aggregate (excluding initial_value)
    const int last_valid_lane = ::cuda::std::clamp(valid_items - 1, 0, LOGICAL_WARP_THREADS - 1);
    warp_aggregate = ::cuda::device::warp_shuffle_idx<LOGICAL_WARP_THREADS>(inclusive, last_valid_lane, member_mask);
    // Update inclusive with initial value and compute exclusive
    UpdatePartial(input, inclusive, exclusive, scan_op, valid_items, initial_value);
  }
};
} // namespace detail

CUB_NAMESPACE_END

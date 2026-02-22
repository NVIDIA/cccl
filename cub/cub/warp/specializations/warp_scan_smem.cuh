// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::WarpScanSmem provides smem-based variants of parallel prefix scan of items partitioned
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

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/__functional/operator_properties.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__utility/static_for.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief WarpScanSmem provides smem-based variants of parallel prefix scan of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being scanned
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp
 */
template <typename T, int LOGICAL_WARP_THREADS>
struct WarpScanSmem
{
  /******************************************************************************
   * Constants and type definitions
   ******************************************************************************/

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == warp_threads);

  /// The number of warp scan steps
  static constexpr int STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE;

  /// The number of threads in half a warp
  static constexpr int HALF_WARP_THREADS = 1 << (STEPS - 1);

  /// The number of shared memory elements per warp
  static constexpr int WARP_SMEM_ELEMENTS = LOGICAL_WARP_THREADS + HALF_WARP_THREADS;

  /// Shared memory storage layout type (1.5 warps-worth of elements for each warp)
  using _TempStorage = T[WARP_SMEM_ELEMENTS];

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  _TempStorage& temp_storage;
  unsigned int lane_id;
  unsigned int member_mask;

  /******************************************************************************
   * Construction
   ******************************************************************************/

  /// Constructor
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpScanSmem(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      ,

      lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : ::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS)
      ,

      member_mask(WarpMask<LOGICAL_WARP_THREADS>(::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
  {}

  /******************************************************************************
   * Utility methods
   ******************************************************************************/

  /// Basic inclusive scan iteration (template unrolled, inductive-case specialization)
  template <bool HAS_IDENTITY, int STEP, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanStep(T& partial, ScanOp scan_op, constant_t<STEP> /*step*/)
  {
    constexpr int OFFSET = 1 << STEP;

    // Share partial into buffer
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], partial);

    __syncwarp(member_mask);

    // Update partial if addend is in range
    if (HAS_IDENTITY || (lane_id >= OFFSET))
    {
      T addend = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - OFFSET]);
      partial  = scan_op(addend, partial);
    }
    __syncwarp(member_mask);

    ScanStep<HAS_IDENTITY>(partial, scan_op, constant_v<STEP + 1>);
  }

  /// Basic inclusive scan iteration(template unrolled, base-case specialization)
  template <bool HAS_IDENTITY, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanStep(T& /*partial*/, ScanOp /*scan_op*/, constant_t<STEPS> /*step*/)
  {}

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
  _CCCL_DEVICE _CCCL_FORCEINLINE T Broadcast(T input, unsigned int src_lane)
  {
    if (lane_id == src_lane)
    {
      ThreadStore<STORE_VOLATILE>(temp_storage, input);
    }

    __syncwarp(member_mask);

    return (T) ThreadLoad<LOAD_VOLATILE>(temp_storage);
  }

  //---------------------------------------------------------------------
  // Inclusive operations
  //---------------------------------------------------------------------

  /**
   * @brief Inclusive scan
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[out] inclusive_output
   *   Calling thread's output item. May be aliased with @p input
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op)
  {
    if constexpr (::cuda::has_identity_element_v<ScanOp, T>)
    {
      constexpr T identity = ::cuda::identity_element<ScanOp, T>();
      ThreadStore<STORE_VOLATILE>(&temp_storage[lane_id], identity);
      __syncwarp(member_mask);
    }

    // Iterate scan steps
    inclusive_output = input;
    ScanStep<::cuda::has_identity_element_v<ScanOp, T>>(inclusive_output, scan_op, constant_v<0>);
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
   *   Warp-wide aggregate reduction of input items.
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op, T& warp_aggregate)
  {
    InclusiveScan(input, inclusive_output, scan_op);

    // Retrieve aggregate
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive_output);

    __syncwarp(member_mask);

    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);

    __syncwarp(member_mask);
  }

  /**
   * @brief Partial inclusive scan
   *
   * @param[in] input
   *   Calling thread's input item.
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
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScanPartial(T input, T& inclusive_output, ScanOp scan_op, int valid_items)
  {
    // Avoid reading uninitialized memory
    // TODO(pauleonix): Is there a cheaper way of ensuring no uninitialized reads?
    cub::detail::uninitialized_copy_single(&temp_storage[lane_id], T{});
    __syncwarp(member_mask);

    // Iterate scan steps
    if (static_cast<int>(lane_id) < valid_items)
    {
      inclusive_output = input;
    }
    ::cuda::static_for<STEPS>([&](auto step) {
      constexpr int offset = 1 << step;

      // Share partial into buffer
      if constexpr (step == 0)
      {
        // Upper half is still uninitialized, i.e. some positions are initialized twice without destructing in between.
        cub::detail::uninitialized_copy_single(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive_output);
      }
      else
      {
        temp_storage[HALF_WARP_THREADS + lane_id] = inclusive_output;
      }
      __syncwarp(member_mask);
      T addend = temp_storage[HALF_WARP_THREADS + lane_id - offset];
      // Update partial if addend is in range
      if ((lane_id >= offset) && (static_cast<int>(lane_id) < valid_items))
      {
        inclusive_output = scan_op(addend, inclusive_output);
      }
      // TODO(pauleonix): Possible optimization: Avoid this in the last iteration.
      // Downside: Need to be very careful syncing in other functions when writing to temp_storage.
      __syncwarp(member_mask);
    });
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
   *   Warp-wide aggregate reduction of input items.
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScanPartial(T input, T& inclusive_output, ScanOp scan_op, int valid_items, T& warp_aggregate)
  {
    InclusiveScanPartial(input, inclusive_output, scan_op, valid_items);

    // Retrieve aggregate
    temp_storage[HALF_WARP_THREADS + lane_id] = inclusive_output;

    __syncwarp(member_mask);

    warp_aggregate = temp_storage[HALF_WARP_THREADS + ::cuda::std::clamp(valid_items - 1, 0, LOGICAL_WARP_THREADS - 1)];

    __syncwarp(member_mask);
  }

  //---------------------------------------------------------------------
  // Get exclusive from inclusive
  //---------------------------------------------------------------------

  /**
   * @brief Update inclusive and exclusive using input and inclusive
   *
   * @param[in] input
   *
   * @param[in, out] inclusive
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
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive);

    __syncwarp(member_mask);

    exclusive = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1]);
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
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive);

    __syncwarp(member_mask);

    exclusive = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1]);
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
  Update(T /*input*/, T& inclusive, T& exclusive, T& warp_aggregate, ScanOpT /*scan_op*/, IsIntegerT /*is_integer*/)
  {
    // Initial value presumed to be unknown or identity (either way our padding is correct)
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive);

    __syncwarp(member_mask);

    exclusive      = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1]);
    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input and inclusive (specialized
   *        for summation of integer types)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T input,
    T& inclusive,
    T& exclusive,
    T& warp_aggregate,
    ::cuda::std::plus<> /*scan_o*/,
    ::cuda::std::true_type /*is_integer*/)
  {
    // Initial value presumed to be unknown or identity (either way our padding is correct)
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive);

    __syncwarp(member_mask);

    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);
    exclusive      = inclusive - input;
  }

  /**
   * @brief Update inclusive, exclusive, and warp aggregate using input, inclusive, and initial
   *        value
   */
  template <typename ScanOpT, typename IsIntegerT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Update(
    T /*input*/,
    T& inclusive,
    T& exclusive,
    T& warp_aggregate,
    ScanOpT scan_op,
    T initial_value,
    IsIntegerT /*is_integer*/)
  {
    // Broadcast warp aggregate
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id], inclusive);

    __syncwarp(member_mask);

    warp_aggregate = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[WARP_SMEM_ELEMENTS - 1]);

    __syncwarp(member_mask);

    // Update inclusive with initial value
    inclusive = scan_op(initial_value, inclusive);

    // Get exclusive from exclusive
    ThreadStore<STORE_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 1], inclusive);

    __syncwarp(member_mask);

    exclusive = (T) ThreadLoad<LOAD_VOLATILE>(&temp_storage[HALF_WARP_THREADS + lane_id - 2]);

    if (lane_id == 0)
    {
      exclusive = initial_value;
    }
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
      temp_storage[HALF_WARP_THREADS + lane_id] = inclusive;

      __syncwarp(member_mask);

      T temp = temp_storage[HALF_WARP_THREADS + lane_id - 1];
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
    if (static_cast<int>(lane_id) < valid_items)
    {
      inclusive = scan_op(initial_value, inclusive);
    }
    // Get exclusive
    UpdatePartial(input, inclusive, exclusive, scan_op, valid_items);
    if constexpr (!(::cuda::std::is_integral_v<T> && cub::detail::is_cuda_std_plus_v<ScanOpT, T>) )
    {
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
  _CCCL_DEVICE _CCCL_FORCEINLINE void UpdatePartial(
    [[maybe_unused]] T input,
    T& inclusive,
    T& exclusive,
    T& warp_aggregate,
    [[maybe_unused]] ScanOpT scan_op,
    int valid_items)
  {
    // Initial value presumed to be unknown or identity (either way our padding is correct)
    temp_storage[HALF_WARP_THREADS + lane_id] = inclusive;

    __syncwarp(member_mask);

    const int last_valid_lane = ::cuda::std::clamp(valid_items - 1, 0, LOGICAL_WARP_THREADS - 1);
    warp_aggregate            = temp_storage[HALF_WARP_THREADS + last_valid_lane];
    // Compute exclusive
    if constexpr (::cuda::std::is_integral_v<T> && cub::detail::is_cuda_std_plus_v<ScanOpT, T>)
    {
      UpdatePartial(input, inclusive, exclusive, scan_op, valid_items);
    }
    else
    {
      // Should not be replaced with UpdatePartial() to avoid redundant store of inclusive and sync.
      T temp = temp_storage[HALF_WARP_THREADS + lane_id - 1];
      if (static_cast<int>(lane_id) < valid_items)
      {
        exclusive = temp;
      }
    }
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
    // Broadcast warp aggregate
    temp_storage[HALF_WARP_THREADS + lane_id] = inclusive;

    __syncwarp(member_mask);

    const int last_valid_lane = ::cuda::std::clamp(valid_items - 1, 0, LOGICAL_WARP_THREADS - 1);
    warp_aggregate            = temp_storage[HALF_WARP_THREADS + last_valid_lane];

    __syncwarp(member_mask);

    // Apply initial_value and get exclusive
    UpdatePartial(input, inclusive, exclusive, scan_op, valid_items, initial_value);
  }
};
} // namespace detail

CUB_NAMESPACE_END

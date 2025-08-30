/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file
 * cub::BlockScanRaking provides variants of raking-based parallel prefix scan across a
 * CUDA thread block.
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

#include <cub/block/block_raking_layout.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_scan.cuh>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief BlockScanRaking provides variants of raking-based parallel prefix scan across a CUDA
 * thread block.
 *
 * @tparam T
 *   Data type being scanned
 *
 * @tparam BLOCK_DIM_X
 *   The thread block length in threads along the X dimension
 *
 * @tparam BLOCK_DIM_Y
 *   The thread block length in threads along the Y dimension
 *
 * @tparam BLOCK_DIM_Z
 *   The thread block length in threads along the Z dimension
 *
 * @tparam MEMOIZE
 *   Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the
 * expense of higher register pressure
 */
template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z, bool MEMOIZE>
struct BlockScanRaking
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  /// The thread block size in threads
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

  /// Layout type for padded thread block raking grid
  using BlockRakingLayout = BlockRakingLayout<T, BLOCK_THREADS>;

  /// Constants
  /// Number of raking threads
  static constexpr int RAKING_THREADS = BlockRakingLayout::RAKING_THREADS;

  /// Number of raking elements per warp synchronous raking thread
  static constexpr int SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH;

  /// Cooperative work can be entirely warp synchronous
  static constexpr bool WARP_SYNCHRONOUS = (BLOCK_THREADS == RAKING_THREADS);

  ///  WarpScan utility type
  using WarpScan = WarpScan<T, RAKING_THREADS>;

  /// Shared memory storage layout type
  struct _TempStorage
  {
    /// Buffer for warp-synchronous scan
    typename WarpScan::TempStorage warp_scan;

    /// Padded thread block raking grid
    typename BlockRakingLayout::TempStorage raking_grid;

    /// Block aggregate
    T block_aggregate;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  // Thread fields
  _TempStorage& temp_storage;
  unsigned int linear_tid;
  T cached_segment[SEGMENT_LENGTH];

  //---------------------------------------------------------------------
  // Utility methods
  //---------------------------------------------------------------------

  /**
   * @brief Templated reduction
   *
   * @param[in] raking_ptr
   *   Input array
   *
   * @param[in] scan_op
   *   Binary reduction operator
   *
   * @param[in] raking_partial
   *   Prefix to seed reduction with
   */
  template <int ITERATION, typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  GuardedReduce(T* raking_ptr, ScanOp scan_op, T raking_partial, constant_t<ITERATION> /*iteration*/)
  {
    if ((BlockRakingLayout::UNGUARDED) || (((linear_tid * SEGMENT_LENGTH) + ITERATION) < BLOCK_THREADS))
    {
      T addend       = raking_ptr[ITERATION];
      raking_partial = scan_op(raking_partial, addend);
    }

    return GuardedReduce(raking_ptr, scan_op, raking_partial, constant_v<ITERATION + 1>);
  }

  /**
   * @brief Templated reduction (base case)
   *
   * @param[in] raking_ptr
   *   Input array
   *
   * @param[in] scan_op
   *   Binary reduction operator
   *
   * @param[in] raking_partial
   *   Prefix to seed reduction with
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  GuardedReduce(T* /*raking_ptr*/, ScanOp /*scan_op*/, T raking_partial, constant_t<SEGMENT_LENGTH> /*iteration*/)
  {
    return raking_partial;
  }

  /**
   * @brief Templated copy
   *
   * @param out
   *   [out] Out array
   *
   * @param in
   *   [in] Input array
   */
  template <int ITERATION>
  _CCCL_DEVICE _CCCL_FORCEINLINE void CopySegment(T* out, T* in, constant_t<ITERATION> /*iteration*/)
  {
    out[ITERATION] = in[ITERATION];
    CopySegment(out, in, constant_v<ITERATION + 1>);
  }

  /**
   * @brief Templated copy (base case)
   *
   * @param[out] out
   *   Out array
   *
   * @param[in] in
   *   Input array
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void CopySegment(T* /*out*/, T* /*in*/, constant_t<SEGMENT_LENGTH> /*iteration*/) {}

  /// Performs upsweep raking reduction, returning the aggregate
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Upsweep(ScanOp scan_op)
  {
    T* smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

    // Read data into registers
    CopySegment(cached_segment, smem_raking_ptr, constant_v<0>);

    T raking_partial = cached_segment[0];

    return GuardedReduce(cached_segment, scan_op, raking_partial, constant_v<1>);
  }

  /// Performs exclusive downsweep raking scan
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveDownsweep(ScanOp scan_op, T raking_partial, bool apply_prefix = true)
  {
    T* smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

    // Read data back into registers
    if (!MEMOIZE)
    {
      CopySegment(cached_segment, smem_raking_ptr, constant_v<0>);
    }

    detail::ThreadScanExclusive(cached_segment, cached_segment, scan_op, raking_partial, apply_prefix);

    // Write data back to smem
    CopySegment(smem_raking_ptr, cached_segment, constant_v<0>);
  }

  /// Performs inclusive downsweep raking scan
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveDownsweep(ScanOp scan_op, T raking_partial, bool apply_prefix = true)
  {
    T* smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

    // Read data back into registers
    if (!MEMOIZE)
    {
      CopySegment(cached_segment, smem_raking_ptr, constant_v<0>);
    }

    detail::ThreadScanInclusive(cached_segment, cached_segment, scan_op, raking_partial, apply_prefix);

    // Write data back to smem
    CopySegment(smem_raking_ptr, cached_segment, constant_v<0>);
  }

  /// Performs upsweep raking reduction, returning the aggregate
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T UpsweepPartialTile(ScanOp scan_op, int segment_valid_items)
  {
    T* const smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

    // Read data into registers
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int iteration = 0; iteration < SEGMENT_LENGTH; ++iteration)
    {
      cached_segment[iteration] = smem_raking_ptr[iteration];
    }

    return cub::detail::ThreadReducePartial(cached_segment, scan_op, segment_valid_items);
  }

  /// Performs exclusive downsweep raking scan
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveDownsweepPartialTile(ScanOp scan_op, T raking_partial, int segment_valid_items, bool apply_prefix = true)
  {
    T* const smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

    // Read data back into registers
    if constexpr (!MEMOIZE)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int iteration = 0; iteration < SEGMENT_LENGTH; ++iteration)
      {
        cached_segment[iteration] = smem_raking_ptr[iteration];
      }
    }

    cub::detail::ThreadScanExclusivePartial(
      cached_segment, cached_segment, scan_op, segment_valid_items, raking_partial, apply_prefix);

    // Write data back to smem
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int iteration = 0; iteration < SEGMENT_LENGTH; ++iteration)
    {
      smem_raking_ptr[iteration] = cached_segment[iteration];
    }
  }

  /// Performs inclusive downsweep raking scan
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveDownsweepPartialTile(ScanOp scan_op, T raking_partial, int segment_valid_items, bool apply_prefix = true)
  {
    T* const smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

    // Read data back into registers
    if constexpr (!MEMOIZE)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int iteration = 0; iteration < SEGMENT_LENGTH; ++iteration)
      {
        cached_segment[iteration] = smem_raking_ptr[iteration];
      }
    }

    cub::detail::ThreadScanInclusivePartial(
      cached_segment, cached_segment, scan_op, segment_valid_items, raking_partial, apply_prefix);

    // Write data back to smem
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int iteration = 0; iteration < SEGMENT_LENGTH; ++iteration)
    {
      smem_raking_ptr[iteration] = cached_segment[iteration];
    }
  }

  //---------------------------------------------------------------------
  // Constructors
  //---------------------------------------------------------------------

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockScanRaking(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  //---------------------------------------------------------------------
  // Exclusive scans
  //---------------------------------------------------------------------

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor. Each thread contributes one input element. With no initial value,
   *        the output computed for <em>thread</em><sub>0</sub> is undefined.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] exclusive_output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScan(input, exclusive_output, scan_op);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Warp-synchronous scan
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan).ExclusiveScan(upsweep_partial, exclusive_partial, scan_op);

        // Exclusive raking downsweep scan
        ExclusiveDownsweep(scan_op, exclusive_partial, (linear_tid != 0));
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      exclusive_output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   * scan_op functor.  Each thread contributes one input element.
   *
   * @param[in] input
   *   Calling thread's input items
   *
   * @param[out] output
   *   Calling thread's output items (may be aliased to \p input)
   *
   * @param[in] initial_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& output, const T& initial_value, ScanOp scan_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScan(input, output, initial_value, scan_op);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Exclusive Warp-synchronous scan
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan).ExclusiveScan(upsweep_partial, exclusive_partial, initial_value, scan_op);

        // Exclusive raking downsweep scan
        ExclusiveDownsweep(scan_op, exclusive_partial);
      }

      __syncthreads();

      // Grab exclusive partial from shared memory
      output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  Also provides every
   *        thread with the block-wide \p block_aggregate of all inputs.  With no initial value,
   *        the output computed for <em>thread</em><sub>0</sub> is undefined.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[out] block_aggregate
   *   Threadblock-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& output, ScanOp scan_op, T& block_aggregate)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScan(input, output, scan_op, block_aggregate);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Warp-synchronous scan
        T inclusive_partial;
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan).Scan(upsweep_partial, inclusive_partial, exclusive_partial, scan_op);

        // Exclusive raking downsweep scan
        ExclusiveDownsweep(scan_op, exclusive_partial, (linear_tid != 0));

        // Broadcast aggregate to all threads
        if (linear_tid == RAKING_THREADS - 1)
        {
          temp_storage.block_aggregate = inclusive_partial;
        }
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;

      // Retrieve block aggregate
      block_aggregate = temp_storage.block_aggregate;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  Also provides every
   *        thread with the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input items
   *
   * @param[out] output
   *   Calling thread's output items (may be aliased to \p input)
   *
   * @param[in] initial_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[out] block_aggregate
   *   Threadblock-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& output, const T& initial_value, ScanOp scan_op, T& block_aggregate)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScan(input, output, initial_value, scan_op, block_aggregate);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Warp-synchronous scan
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ExclusiveScan(upsweep_partial, exclusive_partial, initial_value, scan_op, block_aggregate);

        // Exclusive raking downsweep scan
        ExclusiveDownsweep(scan_op, exclusive_partial);

        // Broadcast aggregate to other threads
        if (linear_tid == 0)
        {
          temp_storage.block_aggregate = block_aggregate;
        }
      }

      __syncthreads();

      // Grab exclusive partial from shared memory
      output = *placement_ptr;

      // Retrieve block aggregate
      block_aggregate = temp_storage.block_aggregate;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  the call-back functor \p
   *        block_prefix_callback_op is invoked by the first warp in the block, and the value
   *        returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that
   *        logically prefixes the thread block's scan inputs.  Also provides every thread with
   *        the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in-out] block_prefix_callback_op
   *   <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread
   *   block-wide prefix to be applied to all inputs.
   */
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      T block_aggregate;
      WarpScan warp_scan(temp_storage.warp_scan);
      warp_scan.ExclusiveScan(input, output, scan_op, block_aggregate);

      // Obtain warp-wide prefix in lane0, then broadcast to other lanes
      T block_prefix = block_prefix_callback_op(block_aggregate);
      block_prefix   = warp_scan.Broadcast(block_prefix, 0);

      output = scan_op(block_prefix, output);
      if (linear_tid == 0)
      {
        output = block_prefix;
      }
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        WarpScan warp_scan(temp_storage.warp_scan);

        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Warp-synchronous scan
        T exclusive_partial, block_aggregate;
        warp_scan.ExclusiveScan(upsweep_partial, exclusive_partial, scan_op, block_aggregate);

        // Obtain block-wide prefix in lane0, then broadcast to other lanes
        T block_prefix = block_prefix_callback_op(block_aggregate);
        block_prefix   = warp_scan.Broadcast(block_prefix, 0);

        // Update prefix with warpscan exclusive partial
        T downsweep_prefix = scan_op(block_prefix, exclusive_partial);
        if (linear_tid == 0)
        {
          downsweep_prefix = block_prefix;
        }

        // Exclusive raking downsweep scan
        ExclusiveDownsweep(scan_op, downsweep_prefix);
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor. Each thread contributes one input element. With no initial value,
   *        the output computed for <em>thread</em><sub>0</sub> is undefined.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] exclusive_output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScanPartialTile(T input, T& exclusive_output, ScanOp scan_op, int valid_items)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScanPartial(input, exclusive_output, scan_op, valid_items);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ExclusiveScanPartial(upsweep_partial, exclusive_partial, scan_op, valid_segments);

        // Exclusive raking downsweep scan
        ExclusiveDownsweepPartialTile(scan_op, exclusive_partial, segment_valid_items, (linear_tid != 0u));
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      exclusive_output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   * scan_op functor.  Each thread contributes one input element.
   *
   * @param[in] input
   *   Calling thread's input items
   *
   * @param[out] output
   *   Calling thread's output items (may be aliased to \p input)
   *
   * @param[in] initial_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScanPartialTile(T input, T& output, const T& initial_value, ScanOp scan_op, int valid_items)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScanPartial(input, output, initial_value, scan_op, valid_items);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Exclusive Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ExclusiveScanPartial(upsweep_partial, exclusive_partial, initial_value, scan_op, valid_segments);

        // Exclusive raking downsweep scan
        ExclusiveDownsweepPartialTile(scan_op, exclusive_partial, segment_valid_items);
      }

      __syncthreads();

      // Grab exclusive partial from shared memory
      output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  Also provides every
   *        thread with the block-wide \p block_aggregate of all inputs.  With no initial value,
   *        the output computed for <em>thread</em><sub>0</sub> is undefined.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   *
   * @param[out] block_aggregate
   *   Threadblock-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScanPartialTile(T input, T& output, ScanOp scan_op, int valid_items, T& block_aggregate)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).ExclusiveScanPartial(input, output, scan_op, valid_items, block_aggregate);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T inclusive_partial;
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ScanPartial(upsweep_partial, inclusive_partial, exclusive_partial, scan_op, valid_segments);

        // Exclusive raking downsweep scan
        ExclusiveDownsweepPartialTile(scan_op, exclusive_partial, segment_valid_items, (linear_tid != 0u));

        // Broadcast aggregate to all threads
        if (static_cast<int>(linear_tid) == ::cuda::std::min(valid_segments - 1, RAKING_THREADS - 1))
        {
          temp_storage.block_aggregate = inclusive_partial;
        }
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;

      // Retrieve block aggregate
      block_aggregate = temp_storage.block_aggregate;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  Also provides every
   *        thread with the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input items
   *
   * @param[out] output
   *   Calling thread's output items (may be aliased to \p input)
   *
   * @param[in] initial_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   *
   * @param[out] block_aggregate
   *   Threadblock-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScanPartialTile(
    T input, T& output, const T& initial_value, ScanOp scan_op, int valid_items, T& block_aggregate)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan)
        .ExclusiveScanPartial(input, output, initial_value, scan_op, valid_items, block_aggregate);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ExclusiveScanPartial(
            upsweep_partial, exclusive_partial, initial_value, scan_op, valid_segments, block_aggregate);

        // Exclusive raking downsweep scan
        ExclusiveDownsweepPartialTile(scan_op, exclusive_partial, segment_valid_items);

        // Broadcast aggregate to other threads
        if (linear_tid == 0u)
        {
          temp_storage.block_aggregate = block_aggregate;
        }
      }

      __syncthreads();

      // Grab exclusive partial from shared memory
      output = *placement_ptr;

      // Retrieve block aggregate
      block_aggregate = temp_storage.block_aggregate;
    }
  }

  /**
   * @brief Computes an exclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  the call-back functor \p
   *        block_prefix_callback_op is invoked by the first warp in the block, and the value
   *        returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that
   *        logically prefixes the thread block's scan inputs.  Also provides every thread with
   *        the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   *
   * @param[in-out] block_prefix_callback_op
   *   <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread
   *   block-wide prefix to be applied to all inputs.
   */
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScanPartialTile(
    T input, T& output, ScanOp scan_op, int valid_items, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      T block_aggregate;
      WarpScan warp_scan(temp_storage.warp_scan);
      warp_scan.ExclusiveScanPartial(input, output, scan_op, valid_items, block_aggregate);

      // Obtain warp-wide prefix in lane0, then broadcast to other lanes
      T block_prefix = block_prefix_callback_op(block_aggregate);
      block_prefix   = warp_scan.Broadcast(block_prefix, 0);

      if (static_cast<int>(linear_tid) < valid_items)
      {
        if (linear_tid == 0u)
        {
          output = block_prefix;
        }
        else
        {
          output = scan_op(block_prefix, output);
        }
      }
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        WarpScan warp_scan(temp_storage.warp_scan);

        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T exclusive_partial;
        T block_aggregate;
        warp_scan.ExclusiveScanPartial(upsweep_partial, exclusive_partial, scan_op, valid_segments, block_aggregate);

        // Obtain block-wide prefix in lane0, then broadcast to other lanes
        T block_prefix = block_prefix_callback_op(block_aggregate);
        block_prefix   = warp_scan.Broadcast(block_prefix, 0);

        // Update prefix with warpscan exclusive partial
        T downsweep_prefix = block_prefix;
        if ((linear_tid != 0u) && (static_cast<int>(linear_tid) < valid_segments))
        {
          downsweep_prefix = scan_op(block_prefix, exclusive_partial);
        }

        // Exclusive raking downsweep scan
        ExclusiveDownsweepPartialTile(scan_op, downsweep_prefix, segment_valid_items);
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;
    }
  }

  //---------------------------------------------------------------------
  // Inclusive scans
  //---------------------------------------------------------------------

  /**
   * @brief Computes an inclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor. Each thread contributes one input element.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& output, ScanOp scan_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).InclusiveScan(input, output, scan_op);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Exclusive Warp-synchronous scan
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan).ExclusiveScan(upsweep_partial, exclusive_partial, scan_op);

        // Inclusive raking downsweep scan
        InclusiveDownsweep(scan_op, exclusive_partial, (linear_tid != 0));
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an inclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor. Each thread contributes one input element.  Also provides every
   *        thread with the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[out] block_aggregate
   *   Threadblock-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& output, ScanOp scan_op, T& block_aggregate)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).InclusiveScan(input, output, scan_op, block_aggregate);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Warp-synchronous scan
        T inclusive_partial;
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan).Scan(upsweep_partial, inclusive_partial, exclusive_partial, scan_op);

        // Inclusive raking downsweep scan
        InclusiveDownsweep(scan_op, exclusive_partial, (linear_tid != 0));

        // Broadcast aggregate to all threads
        if (linear_tid == RAKING_THREADS - 1)
        {
          temp_storage.block_aggregate = inclusive_partial;
        }
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;

      // Retrieve block aggregate
      block_aggregate = temp_storage.block_aggregate;
    }
  }

  /**
   * @brief Computes an inclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  the call-back functor \p
   *        block_prefix_callback_op is invoked by the first warp in the block, and the value
   *        returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that
   *        logically prefixes the thread block's scan inputs.  Also provides every thread with
   *        the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in-out] block_prefix_callback_op
   *   <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread
   *   block-wide prefix to be applied to all inputs.
   */
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T input, T& output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      T block_aggregate;
      WarpScan warp_scan(temp_storage.warp_scan);
      warp_scan.InclusiveScan(input, output, scan_op, block_aggregate);

      // Obtain warp-wide prefix in lane0, then broadcast to other lanes
      T block_prefix = block_prefix_callback_op(block_aggregate);
      block_prefix   = warp_scan.Broadcast(block_prefix, 0);

      // Update prefix with exclusive warpscan partial
      output = scan_op(block_prefix, output);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (linear_tid < RAKING_THREADS)
      {
        WarpScan warp_scan(temp_storage.warp_scan);

        // Raking upsweep reduction across shared partials
        T upsweep_partial = Upsweep(scan_op);

        // Warp-synchronous scan
        T exclusive_partial, block_aggregate;
        warp_scan.ExclusiveScan(upsweep_partial, exclusive_partial, scan_op, block_aggregate);

        // Obtain block-wide prefix in lane0, then broadcast to other lanes
        T block_prefix = block_prefix_callback_op(block_aggregate);
        block_prefix   = warp_scan.Broadcast(block_prefix, 0);

        // Update prefix with warpscan exclusive partial
        T downsweep_prefix = scan_op(block_prefix, exclusive_partial);
        if (linear_tid == 0)
        {
          downsweep_prefix = block_prefix;
        }

        // Inclusive raking downsweep scan
        InclusiveDownsweep(scan_op, downsweep_prefix);
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an inclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor. Each thread contributes one input element.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScanPartialTile(T input, T& output, ScanOp scan_op, int valid_items)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).InclusiveScanPartial(input, output, scan_op, valid_items);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Exclusive Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ExclusiveScanPartial(upsweep_partial, exclusive_partial, scan_op, valid_segments);

        // Inclusive raking downsweep scan
        InclusiveDownsweepPartialTile(scan_op, exclusive_partial, segment_valid_items, (linear_tid != 0u));
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;
    }
  }

  /**
   * @brief Computes an inclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor. Each thread contributes one input element.  Also provides every
   *        thread with the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   *
   * @param[out] block_aggregate
   *   Threadblock-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScanPartialTile(T input, T& output, ScanOp scan_op, int valid_items, T& block_aggregate)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      WarpScan(temp_storage.warp_scan).InclusiveScanPartial(input, output, scan_op, valid_items, block_aggregate);
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T inclusive_partial;
        T exclusive_partial;
        WarpScan(temp_storage.warp_scan)
          .ScanPartial(upsweep_partial, inclusive_partial, exclusive_partial, scan_op, valid_segments);

        // Inclusive raking downsweep scan
        InclusiveDownsweepPartialTile(scan_op, exclusive_partial, segment_valid_items, (linear_tid != 0u));

        // Broadcast aggregate to all threads
        if (static_cast<int>(linear_tid) == ::cuda::std::min(valid_segments - 1, RAKING_THREADS - 1))
        {
          temp_storage.block_aggregate = inclusive_partial;
        }
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;

      // Retrieve block aggregate
      block_aggregate = temp_storage.block_aggregate;
    }
  }

  /**
   * @brief Computes an inclusive thread block-wide prefix scan using the specified binary \p
   *        scan_op functor.  Each thread contributes one input element.  the call-back functor \p
   *        block_prefix_callback_op is invoked by the first warp in the block, and the value
   *        returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that
   *        logically prefixes the thread block's scan inputs.  Also provides every thread with
   *        the block-wide \p block_aggregate of all inputs.
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[out] output
   *   Calling thread's output item (may be aliased to \p input)
   *
   * @param[in] scan_op
   *   Binary scan operator
   *
   * @param[in] valid_items
   *   Number of valid items
   *
   * @param[in-out] block_prefix_callback_op
   *   <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread
   *   block-wide prefix to be applied to all inputs.
   */
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScanPartialTile(
    T input, T& output, ScanOp scan_op, int valid_items, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if constexpr (WARP_SYNCHRONOUS)
    {
      // Short-circuit directly to warp-synchronous scan
      T block_aggregate;
      WarpScan warp_scan(temp_storage.warp_scan);
      warp_scan.InclusiveScanPartial(input, output, scan_op, valid_items, block_aggregate);

      // Obtain warp-wide prefix in lane0, then broadcast to other lanes
      T block_prefix = block_prefix_callback_op(block_aggregate);
      block_prefix   = warp_scan.Broadcast(block_prefix, 0);

      // Update prefix with exclusive warpscan partial
      if (static_cast<int>(linear_tid) < valid_items)
      {
        output = scan_op(block_prefix, output);
      }
    }
    else
    {
      // Place thread partial into shared memory raking grid
      T* const placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
      cub::detail::uninitialized_copy_single(placement_ptr, input);

      __syncthreads();

      // Reduce parallelism down to just raking threads
      if (static_cast<int>(linear_tid) < RAKING_THREADS)
      {
        WarpScan warp_scan(temp_storage.warp_scan);

        // Raking upsweep reduction across shared partials
        int segment_valid_items = valid_items - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
        if constexpr (!BlockRakingLayout::UNGUARDED)
        {
          const int raking_segment_valid_items = BLOCK_THREADS - static_cast<int>(linear_tid) * SEGMENT_LENGTH;
          segment_valid_items                  = ::cuda::std::min(segment_valid_items, raking_segment_valid_items);
        }
        const T upsweep_partial = UpsweepPartialTile(scan_op, segment_valid_items);

        // Warp-synchronous scan
        const int valid_segments = ::cuda::ceil_div(::cuda::std::max(valid_items, 0), SEGMENT_LENGTH);
        T exclusive_partial;
        T block_aggregate;
        warp_scan.ExclusiveScanPartial(upsweep_partial, exclusive_partial, scan_op, valid_segments, block_aggregate);

        // Obtain block-wide prefix in lane0, then broadcast to other lanes
        T block_prefix = block_prefix_callback_op(block_aggregate);
        block_prefix   = warp_scan.Broadcast(block_prefix, 0);

        // Update prefix with warpscan exclusive partial
        T downsweep_prefix = block_prefix;
        if ((linear_tid != 0u) && (static_cast<int>(linear_tid) < valid_segments))
        {
          downsweep_prefix = scan_op(block_prefix, exclusive_partial);
        }

        // Inclusive raking downsweep scan
        InclusiveDownsweepPartialTile(scan_op, downsweep_prefix, segment_valid_items);
      }

      __syncthreads();

      // Grab thread prefix from shared memory
      output = *placement_ptr;
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END

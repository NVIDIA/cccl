// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::detail::BlockScanWarpReduceThenScan provides a warp-granularity "reduce-then-scan" parallel
 * prefix scan across a CUDA thread block, optimized for the latency until the block aggregate (and therefore a block
 * prefix callback) becomes available.
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

#include <cub/block/specializations/block_scan_warp_scans.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/warp/specializations/warp_redux.cuh>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief BlockScanWarpReduceThenScan provides a warp-granularity reduce-then-scan variant of
 *        the warpscans-based parallel prefix scan across a CUDA thread block: each warp first
 *        REDUCES its inputs to the warp aggregate (hardware `redux.sync` where supported),
 *        the aggregates are folded into warp prefixes, and only then do the warps scan.
 *
 * In the classic warpscans algorithm the warp aggregate is the *last* output of each warp's
 * shuffle-scan chain, so the block-wide barrier releases only after the slowest warp's full
 * scan, the cross-warp aggregate fold runs as a serial tail, and a block prefix callback (e.g.
 * the decoupled look-back's partial promotion) is invoked last. When the (scan_op, T) pair is
 * supported by the `redux.sync` instruction (see `is_warp_redux_op_supported`), the warp
 * aggregate can instead be produced in a single instruction *before* the warp scans: the
 * barrier releases as soon as the reductions complete, the aggregate fold overlaps the shuffle
 * chains, and the prefix callback is invoked by warp 0 while the other warps are still
 * scanning — its (global-memory) latency hides in their shadow.
 *
 * Results are identical to BlockScanWarpScans: eligibility is restricted to associative,
 * commutative integer operators for which the redux-computed aggregate is bit-exact.
 * For unsupported operators/types, and on architectures without `redux.sync` (< sm_80), all
 * methods compile to / fall back to the classic BlockScanWarpScans code paths.
 *
 * @tparam BlockDimX
 *   The thread block length in threads along the X dimension
 *
 * @tparam BlockDimY
 *   The thread block length in threads along the Y dimension
 *
 * @tparam BlockDimZ
 *   The thread block length in threads along the Z dimension
 */
template <typename T, int BlockDimX, int BlockDimY, int BlockDimZ>
struct BlockScanWarpReduceThenScan : BlockScanWarpScans<T, BlockDimX, BlockDimY, BlockDimZ>
{
  using Base = BlockScanWarpScans<T, BlockDimX, BlockDimY, BlockDimZ>;
  using Base::BLOCK_THREADS;
  using Base::WARP_THREADS;
  using typename Base::TempStorage;
  using WarpScanT = typename Base::WarpScanT;

  static_assert(BlockDimX * BlockDimY * BlockDimZ % warp_threads == 0,
                "BlockScanWarpReduceThenScan requires the block size to be a multiple of the warp size "
                "(enforced by BlockScan's SAFE_ALGORITHM fallback)");

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockScanWarpReduceThenScan(TempStorage& temp_storage)
      : Base(temp_storage)
  {}

  //---------------------------------------------------------------------
  // Reduce-then-scan core
  //---------------------------------------------------------------------

  /**
   * @brief Produces this warp's aggregate via `redux.sync`, publishes it, and folds all warp
   *        aggregates into the calling warp's prefix and the block aggregate — all BEFORE any
   *        warp scan has run. Returns false when `redux.sync` is unavailable at run time
   *        (pre-sm_80 code paths), in which case the caller must take the classic path.
   *
   * @param[out] warp_prefix  Prefix of all preceding warps' aggregates; valid for warp_id != 0
   * @param[out] block_aggregate  Block-wide aggregate reduction of input items
   */
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE bool
  ReduceThenFoldAggregates(T input, ScanOp scan_op, T& warp_prefix, T& block_aggregate)
  {
    if (const auto warp_aggregate = detail::warp_redux(input, 0xffffffffu, scan_op))
    {
      if (this->lane_id == 0)
      {
        detail::uninitialized_copy_single(this->temp_storage.warp_aggregates + this->warp_id, *warp_aggregate);
      }
      __syncthreads(); // releases as soon as the reductions complete, not the scans

      block_aggregate = this->temp_storage.warp_aggregates[0];
      this->ApplyWarpAggregates(warp_prefix, scan_op, block_aggregate, constant_v<1>);
      return true;
    }
    return false;
  }

  //---------------------------------------------------------------------
  // Exclusive scans
  //---------------------------------------------------------------------

  /// Computes an exclusive thread block-wide prefix scan. With no initial value, the output
  /// computed for thread0 is undefined.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op)
  {
    T block_aggregate;
    ExclusiveScan(input, exclusive_output, scan_op, block_aggregate);
  }

  /// Computes an exclusive thread block-wide prefix scan seeded with @p initial_value.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, const T& initial_value, ScanOp scan_op)
  {
    T block_aggregate;
    ExclusiveScan(input, exclusive_output, initial_value, scan_op, block_aggregate);
  }

  /// Computes an exclusive thread block-wide prefix scan; also provides every thread with the
  /// block-wide @p block_aggregate of all inputs. With no initial value, the output computed
  /// for thread0 is undefined.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op, T& block_aggregate)
  {
    if constexpr (detail::is_warp_redux_op_supported<ScanOp, T>)
    {
      T warp_prefix;
      if (ReduceThenFoldAggregates(input, scan_op, warp_prefix, block_aggregate))
      {
        T inclusive_output;
        WarpScanT(this->temp_storage.warp_scan[this->warp_id]).Scan(input, inclusive_output, exclusive_output, scan_op);
        if (this->warp_id != 0)
        {
          exclusive_output = scan_op(warp_prefix, exclusive_output);
          if (this->lane_id == 0)
          {
            exclusive_output = warp_prefix;
          }
        }
        return;
      }
    }
    Base::ExclusiveScan(input, exclusive_output, scan_op, block_aggregate);
  }

  /// Computes an exclusive thread block-wide prefix scan seeded with @p initial_value; also
  /// provides every thread with the block-wide @p block_aggregate of all inputs.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& exclusive_output, const T& initial_value, ScanOp scan_op, T& block_aggregate)
  {
    if constexpr (detail::is_warp_redux_op_supported<ScanOp, T>)
    {
      T warp_prefix;
      if (ReduceThenFoldAggregates(input, scan_op, warp_prefix, block_aggregate))
      {
        const T prefix = (this->warp_id == 0) ? initial_value : scan_op(initial_value, warp_prefix);

        T inclusive_output;
        WarpScanT(this->temp_storage.warp_scan[this->warp_id]).Scan(input, inclusive_output, exclusive_output, scan_op);
        exclusive_output = scan_op(prefix, exclusive_output);
        if (this->lane_id == 0)
        {
          exclusive_output = prefix;
        }
        return;
      }
    }
    Base::ExclusiveScan(input, exclusive_output, initial_value, scan_op, block_aggregate);
  }

  /// Computes an exclusive thread block-wide prefix scan; the call-back functor is invoked by
  /// the first warp with the block aggregate, and the value returned by lane0 seeds the scan.
  /// This is the aggregate-first sweet spot: the callback (e.g. a decoupled look-back) runs one
  /// warp-scan earlier and its latency overlaps all warps' shuffle scans.
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ExclusiveScan(T input, T& exclusive_output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if constexpr (detail::is_warp_redux_op_supported<ScanOp, T>)
    {
      T warp_prefix;
      T block_aggregate;
      if (ReduceThenFoldAggregates(input, scan_op, warp_prefix, block_aggregate))
      {
        // warp 0 runs the callback immediately; warps 1..N scan in its shadow
        if (this->warp_id == 0)
        {
          T block_prefix = block_prefix_callback_op(block_aggregate);
          if (this->lane_id == 0)
          {
            detail::uninitialized_copy_single(&this->temp_storage.block_prefix, block_prefix);
          }
        }

        T inclusive_output;
        WarpScanT(this->temp_storage.warp_scan[this->warp_id]).Scan(input, inclusive_output, exclusive_output, scan_op);
        __syncthreads();

        const T block_prefix = this->temp_storage.block_prefix;
        const T prefix       = (this->warp_id == 0) ? block_prefix : scan_op(block_prefix, warp_prefix);
        exclusive_output     = scan_op(prefix, exclusive_output);
        if (this->lane_id == 0)
        {
          exclusive_output = prefix;
        }
        return;
      }
    }
    Base::ExclusiveScan(input, exclusive_output, scan_op, block_prefix_callback_op);
  }

  //---------------------------------------------------------------------
  // Inclusive scans
  //---------------------------------------------------------------------

  /// Computes an inclusive thread block-wide prefix scan.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op)
  {
    T block_aggregate;
    InclusiveScan(input, inclusive_output, scan_op, block_aggregate);
  }

  /// Computes an inclusive thread block-wide prefix scan; also provides every thread with the
  /// block-wide @p block_aggregate of all inputs.
  template <typename ScanOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InclusiveScan(T input, T& inclusive_output, ScanOp scan_op, T& block_aggregate)
  {
    if constexpr (detail::is_warp_redux_op_supported<ScanOp, T>)
    {
      T warp_prefix;
      if (ReduceThenFoldAggregates(input, scan_op, warp_prefix, block_aggregate))
      {
        WarpScanT(this->temp_storage.warp_scan[this->warp_id]).InclusiveScan(input, inclusive_output, scan_op);
        if (this->warp_id != 0)
        {
          inclusive_output = scan_op(warp_prefix, inclusive_output);
        }
        return;
      }
    }
    Base::InclusiveScan(input, inclusive_output, scan_op, block_aggregate);
  }

  /// Computes an inclusive thread block-wide prefix scan; the call-back functor is invoked by
  /// the first warp with the block aggregate, and the value returned by lane0 seeds the scan.
  /// See the exclusive callback overload for the aggregate-first rationale.
  template <typename ScanOp, typename BlockPrefixCallbackOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  InclusiveScan(T input, T& exclusive_output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
  {
    if constexpr (detail::is_warp_redux_op_supported<ScanOp, T>)
    {
      T warp_prefix;
      T block_aggregate;
      if (ReduceThenFoldAggregates(input, scan_op, warp_prefix, block_aggregate))
      {
        if (this->warp_id == 0)
        {
          T block_prefix = block_prefix_callback_op(block_aggregate);
          if (this->lane_id == 0)
          {
            detail::uninitialized_copy_single(&this->temp_storage.block_prefix, block_prefix);
          }
        }

        T warp_inclusive;
        WarpScanT(this->temp_storage.warp_scan[this->warp_id]).InclusiveScan(input, warp_inclusive, scan_op);
        __syncthreads();

        const T block_prefix = this->temp_storage.block_prefix;
        const T prefix       = (this->warp_id == 0) ? block_prefix : scan_op(block_prefix, warp_prefix);
        exclusive_output     = scan_op(prefix, warp_inclusive);
        return;
      }
    }
    Base::InclusiveScan(input, exclusive_output, scan_op, block_prefix_callback_op);
  }
};
} // namespace detail

CUB_NAMESPACE_END

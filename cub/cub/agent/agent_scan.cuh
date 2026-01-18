// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file cub::AgentScan implements a stateful abstraction of CUDA thread blocks
 *       for participating in device-wide prefix scan .
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

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_device.cuh>

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
#  include <cub/agent/agent_unique_by_key.cuh> // for UniqueByKeyAgentPolicy
#endif

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * @brief Parameterizable tuning policy type for AgentScan
 *
 * @tparam NominalBlockThreads4B
 *   Threads per thread block
 *
 * @tparam NominalItemsPerThread4B
 *   Items per thread (per tile of input)
 *
 * @tparam ComputeT
 *   Dominant compute type
 *
 * @tparam LoadAlgorithm
 *   The BlockLoad algorithm to use
 *
 * @tparam LoadModifier
 *   Cache load modifier for reading input elements
 *
 * @tparam StoreAlgorithm
 *   The BlockStore algorithm to use
 *
 * @tparam ScanAlgorithm
 *   The BlockScan algorithm to use
 *
 * @tparam DelayConstructorT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <int NominalBlockThreads4B,
          int NominalItemsPerThread4B,
          typename ComputeT,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          BlockStoreAlgorithm StoreAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          typename ScalingType = detail::MemBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>,
          typename DelayConstructorT = detail::default_delay_constructor_t<ComputeT>>
struct AgentScanPolicy : ScalingType
{
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM   = LoadAlgorithm;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = LoadModifier;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = StoreAlgorithm;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM   = ScanAlgorithm;

  struct detail
  {
    using delay_constructor_t = DelayConstructorT;
  };
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
namespace detail
{
// Only define this when needed.
// Because of overload woes, this depends on C++20 concepts. util_device.h checks that concepts are available when
// either runtime policies or PTX JSON information are enabled, so if they are, this is always valid. The generic
// version is always defined, and that's the only one needed for regular CUB operations.
//
// TODO: enable this unconditionally once concepts are always available
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  ScanAgentPolicy,
  (UniqueByKeyAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (LOAD_ALGORITHM, LoadAlgorithm, cub::BlockLoadAlgorithm),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier),
  (STORE_ALGORITHM, StoreAlgorithm, cub::BlockStoreAlgorithm),
  (SCAN_ALGORITHM, ScanAlgorithm, cub::BlockScanAlgorithm))
} // namespace detail
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail::scan
{
/**
 * @brief AgentScan implements a stateful abstraction of CUDA thread blocks for
 *        participating in device-wide prefix scan.
 * @tparam AgentScanPolicyT
 *   Parameterized AgentScanPolicyT tuning policy type
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam ScanOpT
 *   Scan functor type
 *
 * @tparam InitValueT
 *   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 */
template <typename AgentScanPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          bool ForceInclusive = false,
          bool UsePDL         = false>
struct AgentScan
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  using InputT = cub::detail::it_value_t<InputIteratorT>;

  // Tile status descriptor interface type
  using ScanTileStateT = ScanTileState<AccumT>;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<::cuda::std::is_pointer_v<InputIteratorT>,
                     CacheModifiedInputIterator<AgentScanPolicyT::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  // Inclusive scan if no init_value type is provided
  static constexpr bool HAS_INIT     = !::cuda::std::is_same_v<InitValueT, NullType>;
  static constexpr bool IS_INCLUSIVE = ForceInclusive || !HAS_INIT; // We are relying on either initial value not being
                                                                    // `NullType` or the ForceInclusive tag to be true
                                                                    // for inclusive scan to get picked up.
  static constexpr int BLOCK_THREADS    = AgentScanPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = AgentScanPolicyT::ITEMS_PER_THREAD;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Parameterized BlockLoad type
  using BlockLoadT =
    BlockLoad<AccumT,
              AgentScanPolicyT::BLOCK_THREADS,
              AgentScanPolicyT::ITEMS_PER_THREAD,
              AgentScanPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockStore type
  using BlockStoreT =
    BlockStore<AccumT,
               AgentScanPolicyT::BLOCK_THREADS,
               AgentScanPolicyT::ITEMS_PER_THREAD,
               AgentScanPolicyT::STORE_ALGORITHM>;

  // Parameterized BlockScan type
  using BlockScanT = BlockScan<AccumT, AgentScanPolicyT::BLOCK_THREADS, AgentScanPolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using DelayConstructorT     = typename AgentScanPolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackOpT = TilePrefixCallbackOp<AccumT, ScanOpT, ScanTileStateT, DelayConstructorT>;

  // Stateful BlockScan prefix callback type for managing a running total while
  // scanning consecutive tiles
  using RunningPrefixCallbackOp = BlockScanRunningPrefixOp<AccumT, ScanOpT>;

  // Shared memory type for this thread block
  union _TempStorage
  {
    // Smem needed for tile loading
    typename BlockLoadT::TempStorage load;

    // Smem needed for tile storing
    typename BlockStoreT::TempStorage store;

    struct ScanStorage
    {
      // Smem needed for cooperative prefix callback
      typename TilePrefixCallbackOpT::TempStorage prefix;

      // Smem needed for tile scanning
      typename BlockScanT::TempStorage scan;
    } scan_storage;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage; ///< Reference to temp_storage
  WrappedInputIteratorT d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary scan operator
  InitValueT init_value; ///< The init_value element for ScanOpT

  //---------------------------------------------------------------------
  // Block scan utility methods
  //---------------------------------------------------------------------

  template <bool Inclusive = IS_INCLUSIVE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScanFirstTile(AccumT (&items)[ITEMS_PER_THREAD], InitValueT init_value, ScanOpT scan_op, AccumT& block_aggregate)
  {
    BlockScanT blockScan(temp_storage.scan_storage.scan);
    if constexpr (Inclusive)
    {
      if constexpr (HAS_INIT)
      {
        blockScan.InclusiveScan(items, items, init_value, scan_op, block_aggregate);
        block_aggregate = scan_op(init_value, block_aggregate);
      }
      else
      {
        blockScan.InclusiveScan(items, items, scan_op, block_aggregate);
      }
    }
    else
    {
      blockScan.ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
      block_aggregate = scan_op(init_value, block_aggregate);
    }
  }

  template <typename PrefixCallback, bool Inclusive = IS_INCLUSIVE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScanSubsequentTile(AccumT (&items)[ITEMS_PER_THREAD], ScanOpT scan_op, PrefixCallback& prefix_op)
  {
    BlockScanT blockScan(temp_storage.scan_storage.scan);
    if constexpr (Inclusive)
    {
      blockScan.InclusiveScan(items, items, scan_op, prefix_op);
    }
    else
    {
      blockScan.ExclusiveScan(items, items, scan_op, prefix_op);
    }
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_in
   *   Input data
   *
   * @param d_out
   *   Output data
   *
   * @param scan_op
   *   Binary scan operator
   *
   * @param init_value
   *   Initial value to seed the exclusive scan
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentScan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT init_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
  {}

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  /**
   * Process a tile of input (dynamic chained scan)
   * @tparam IS_LAST_TILE
   *   Whether the current tile is the last tile
   *
   * @param num_remaining
   *   Number of global input items remaining (including this tile)
   *
   * @param tile_idx
   *   Tile index
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state
   *   Global tile state descriptor
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(OffsetT num_remaining, int tile_idx, OffsetT tile_offset, ScanTileStateT& tile_state)
  {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if constexpr (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, num_remaining, *(d_in + tile_offset));
    }
    else
    {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    __syncthreads();

    // Perform tile scan
    if (tile_idx == 0)
    {
      // Scan first tile
      AccumT block_aggregate;
      ScanFirstTile(items, init_value, scan_op, block_aggregate);

      if ((!IS_LAST_TILE) && (threadIdx.x == 0))
      {
        tile_state.SetInclusive(0, block_aggregate);
      }
    }
    else
    {
      // Scan non-first tile
      TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.scan_storage.prefix, scan_op, tile_idx);
      ScanSubsequentTile(items, scan_op, prefix_op);
    }

    __syncthreads();

    if constexpr (UsePDL)
    {
      _CCCL_PDL_TRIGGER_NEXT_LAUNCH(); // omitting makes almost no difference in cub.bench.scan.exclusive.sum.base
    }

    // Store items
    if constexpr (IS_LAST_TILE)
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, num_remaining);
    }
    else
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  /**
   * @brief Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_items
   *   Total number of input items
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @param start_tile
   *   The starting tile for the current grid
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT num_items, ScanTileStateT& tile_state, int start_tile)
  {
    // Blocks are launched in increasing order, so just assign one tile per
    // block

    // Current tile index
    int tile_idx = start_tile + blockIdx.x;

    // Global offset for the current tile
    OffsetT tile_offset = OffsetT(TILE_ITEMS) * tile_idx;

    // Remaining items (including this tile)
    OffsetT num_remaining = num_items - tile_offset;

    if (num_remaining > TILE_ITEMS)
    {
      // Not last tile
      ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
    }
    else if (num_remaining > 0)
    {
      // Last tile
      ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
    }
  }

  //---------------------------------------------------------------------------
  // Scan an sequence of consecutive tiles (independent of other thread blocks)
  //---------------------------------------------------------------------------

  /**
   * @brief Process a tile of input
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param prefix_op
   *   Running prefix operator
   *
   * @param valid_items
   *   Number of valid items in the tile
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(OffsetT tile_offset, RunningPrefixCallbackOp& prefix_op, int valid_items = TILE_ITEMS)
  {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if constexpr (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, valid_items, *(d_in + tile_offset));
    }
    else
    {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    __syncthreads();

    // Block scan
    if constexpr (IS_FIRST_TILE)
    {
      AccumT block_aggregate;
      ScanFirstTile(items, init_value, scan_op, block_aggregate);
      prefix_op.running_total = block_aggregate;
    }
    else
    {
      ScanSubsequentTile(items, scan_op, prefix_op);
    }

    __syncthreads();

    // Store items
    if constexpr (IS_LAST_TILE)
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, valid_items);
    }
    else
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  /**
   * @brief Scan a consecutive share of input tiles
   *
   * @param[in] range_offset
   *   Threadblock begin offset (inclusive)
   *
   * @param[in] range_end
   *   Threadblock end offset (exclusive)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT range_offset, OffsetT range_end)
  {
    BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(scan_op);

    if (range_offset + TILE_ITEMS <= range_end)
    {
      // Consume first tile of input (full)
      ConsumeTile<true, true>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;

      // Consume subsequent full tiles of input
      while (range_offset + TILE_ITEMS <= range_end)
      {
        ConsumeTile<false, true>(range_offset, prefix_op);
        range_offset += TILE_ITEMS;
      }

      // Consume a partially-full tile
      if (range_offset < range_end)
      {
        int valid_items = range_end - range_offset;
        ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
      }
    }
    else
    {
      // Consume the first tile of input (partially-full)
      int valid_items = range_end - range_offset;
      ConsumeTile<true, false>(range_offset, prefix_op, valid_items);
    }
  }

  /**
   * @brief Scan a consecutive share of input tiles, seeded with the
   *        specified prefix value
   * @param[in] range_offset
   *   Threadblock begin offset (inclusive)
   *
   * @param[in] range_end
   *   Threadblock end offset (exclusive)
   *
   * @param[in] prefix
   *   The prefix to apply to the scan segment
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT range_offset, OffsetT range_end, AccumT prefix)
  {
    BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(prefix, scan_op);

    // Consume full tiles of input
    while (range_offset + TILE_ITEMS <= range_end)
    {
      ConsumeTile<true, false>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;
    }

    // Consume a partially-full tile
    if (range_offset < range_end)
    {
      int valid_items = range_end - range_offset;
      ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
    }
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END

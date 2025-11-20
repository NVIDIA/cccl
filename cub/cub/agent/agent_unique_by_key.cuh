// SPDX-FileCopyrightText: Copyright (c), NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * cub::AgentUniqueByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide
 * unique-by-key.
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
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/thread/thread_operators.cuh>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentUniqueByKey
 *
 * @tparam DelayConstructorT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <int BlockThreads,
          int ItemsPerThread                    = 1,
          cub::BlockLoadAlgorithm LoadAlgorithm = cub::BLOCK_LOAD_DIRECT,
          cub::CacheLoadModifier LoadModifier   = cub::LOAD_LDG,
          cub::BlockScanAlgorithm ScanAlgorithm = cub::BLOCK_SCAN_WARP_SCANS,
          typename DelayConstructorT            = detail::fixed_delay_constructor_t<350, 450>>
struct AgentUniqueByKeyPolicy
{
  static constexpr int BLOCK_THREADS                      = BlockThreads;
  static constexpr int ITEMS_PER_THREAD                   = ItemsPerThread;
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = LoadAlgorithm;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER   = LoadModifier;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;

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
  UniqueByKeyAgentPolicy,
  (GenericAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (LOAD_ALGORITHM, LoadAlgorithm, cub::BlockLoadAlgorithm),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier),
  (SCAN_ALGORITHM, ScanAlgorithm, cub::BlockScanAlgorithm))
} // namespace detail
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail::unique_by_key
{
/**
 * @brief AgentUniqueByKey implements a stateful abstraction of CUDA thread blocks for participating
 * in device-wide unique-by-key
 *
 * @tparam AgentUniqueByKeyPolicyT
 *   Parameterized AgentUniqueByKeyPolicy tuning policy type
 *
 * @tparam KeyInputIteratorT
 *   Random-access input iterator type for keys
 *
 * @tparam ValueInputIteratorT
 *   Random-access input iterator type for values
 *
 * @tparam KeyOutputIteratorT
 *   Random-access output iterator type for keys
 *
 * @tparam ValueOutputIteratorT
 *   Random-access output iterator type for values
 *
 * @tparam EqualityOpT
 *   Equality operator type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename AgentUniqueByKeyPolicyT,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename EqualityOpT,
          typename OffsetT>
struct AgentUniqueByKey
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input key and value type
  using KeyT   = cub::detail::it_value_t<KeyInputIteratorT>;
  using ValueT = cub::detail::it_value_t<ValueInputIteratorT>;

  // Tile status descriptor interface type
  using ScanTileStateT = ScanTileState<OffsetT>;

  // Constants
  static constexpr int BLOCK_THREADS    = AgentUniqueByKeyPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = AgentUniqueByKeyPolicyT::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Cache-modified Input iterator wrapper type (for applying cache modifier) for keys
  using WrappedKeyInputIteratorT = ::cuda::std::conditional_t<
    ::cuda::std::is_pointer_v<KeyInputIteratorT>,
    CacheModifiedInputIterator<AgentUniqueByKeyPolicyT::LOAD_MODIFIER, KeyT, OffsetT>, // Wrap the native input pointer
                                                                                       // with
                                                                                       // CacheModifiedValuesInputIterator
    KeyInputIteratorT>; // Directly use the supplied input iterator type

  // Cache-modified Input iterator wrapper type (for applying cache modifier) for values
  using WrappedValueInputIteratorT = ::cuda::std::conditional_t<
    ::cuda::std::is_pointer_v<ValueInputIteratorT>,
    CacheModifiedInputIterator<AgentUniqueByKeyPolicyT::LOAD_MODIFIER, ValueT, OffsetT>, // Wrap the native input
                                                                                         // pointer with
                                                                                         // CacheModifiedValuesInputIterator
    ValueInputIteratorT>; // Directly use the supplied input iterator type

  // Parameterized BlockLoad type for input data
  using BlockLoadKeys = BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentUniqueByKeyPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockLoad type for flags
  using BlockLoadValues = BlockLoad<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentUniqueByKeyPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockDiscontinuity type for items
  using BlockDiscontinuityKeys = cub::BlockDiscontinuity<KeyT, BLOCK_THREADS>;

  // Parameterized BlockScan type
  using BlockScanT = cub::BlockScan<OffsetT, BLOCK_THREADS, AgentUniqueByKeyPolicyT::SCAN_ALGORITHM>;

  // Parameterized BlockDiscontinuity type for items
  using DelayConstructorT  = typename AgentUniqueByKeyPolicyT::detail::delay_constructor_t;
  using TilePrefixCallback = cub::TilePrefixCallbackOp<OffsetT, ::cuda::std::plus<>, ScanTileStateT, DelayConstructorT>;

  // Key exchange type
  using KeyExchangeT = KeyT[ITEMS_PER_TILE];

  // Value exchange type
  using ValueExchangeT = ValueT[ITEMS_PER_TILE];

  // Shared memory type for this thread block
  union _TempStorage
  {
    struct ScanStorage
    {
      typename BlockScanT::TempStorage scan;
      typename TilePrefixCallback::TempStorage prefix;
      typename BlockDiscontinuityKeys::TempStorage discontinuity;
    } scan_storage;

    // Smem needed for loading keys
    typename BlockLoadKeys::TempStorage load_keys;

    // Smem needed for loading values
    typename BlockLoadValues::TempStorage load_values;

    // Smem needed for compacting items (allows non POD items in this union)
    Uninitialized<KeyExchangeT> shared_keys;
    Uninitialized<ValueExchangeT> shared_values;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage;
  WrappedKeyInputIteratorT d_keys_in;
  WrappedValueInputIteratorT d_values_in;
  KeyOutputIteratorT d_keys_out;
  ValueOutputIteratorT d_values_out;
  cub::InequalityWrapper<EqualityOpT> inequality_op;
  OffsetT num_items;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentUniqueByKey(
    TempStorage& temp_storage_,
    WrappedKeyInputIteratorT d_keys_in_,
    WrappedValueInputIteratorT d_values_in_,
    KeyOutputIteratorT d_keys_out_,
    ValueOutputIteratorT d_values_out_,
    EqualityOpT equality_op_,
    OffsetT num_items_)
      : temp_storage(temp_storage_.Alias())
      , d_keys_in(d_keys_in_)
      , d_values_in(d_values_in_)
      , d_keys_out(d_keys_out_)
      , d_values_out(d_values_out_)
      , inequality_op(equality_op_)
      , num_items(num_items_)
  {}

  //---------------------------------------------------------------------
  // Utility functions
  //---------------------------------------------------------------------

  struct KeyTagT
  {};
  struct ValueTagT
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE KeyExchangeT& GetShared(KeyTagT)
  {
    return temp_storage.shared_keys.Alias();
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE ValueExchangeT& GetShared(ValueTagT)
  {
    return temp_storage.shared_values.Alias();
  }

  //---------------------------------------------------------------------
  // Scatter utility methods
  //---------------------------------------------------------------------
  template <typename Tag, typename OutputIt, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Scatter(
    Tag tag,
    OutputIt items_out,
    T (&items)[ITEMS_PER_THREAD],
    OffsetT (&selection_flags)[ITEMS_PER_THREAD],
    OffsetT (&selection_indices)[ITEMS_PER_THREAD],
    int /*num_tile_items*/,
    int num_tile_selections,
    OffsetT num_selections_prefix,
    OffsetT /*num_selections*/)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int local_scatter_offset = selection_indices[ITEM] - num_selections_prefix;
      if (selection_flags[ITEM])
      {
        GetShared(tag)[local_scatter_offset] = items[ITEM];
      }
    }

    __syncthreads();

    // Preventing loop unrolling helps avoid perf degradation when switching from signed to unsigned 32-bit offset
    // types
    _CCCL_PRAGMA_NOUNROLL()
    for (int item = threadIdx.x; item < num_tile_selections; item += BLOCK_THREADS)
    {
      items_out[num_selections_prefix + item] = GetShared(tag)[item];
    }

    __syncthreads();
  }

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  /**
   * @brief Process first tile of input (dynamic chained scan).
   *
   * @param num_tile_items
   *   Number of input items comprising this tile
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @return The running count of selections (including this tile)
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
  ConsumeFirstTile(int num_tile_items, OffsetT tile_offset, ScanTileStateT& tile_state)
  {
    KeyT keys[ITEMS_PER_THREAD];
    OffsetT selection_flags[ITEMS_PER_THREAD];
    OffsetT selection_idx[ITEMS_PER_THREAD];

    if constexpr (IS_LAST_TILE)
    {
      // Fill last elements with the first element
      // because collectives are not suffix guarded
      BlockLoadKeys(temp_storage.load_keys)
        .Load(d_keys_in + tile_offset, keys, num_tile_items, *(d_keys_in + tile_offset));
    }
    else
    {
      BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);
    }

    __syncthreads();

    ValueT values[ITEMS_PER_THREAD];
    if constexpr (IS_LAST_TILE)
    {
      // Fill last elements with the first element
      // because collectives are not suffix guarded
      BlockLoadValues(temp_storage.load_values)
        .Load(d_values_in + tile_offset, values, num_tile_items, *(d_values_in + tile_offset));
    }
    else
    {
      BlockLoadValues(temp_storage.load_values).Load(d_values_in + tile_offset, values);
    }

    __syncthreads();

    BlockDiscontinuityKeys(temp_storage.scan_storage.discontinuity).FlagHeads(selection_flags, keys, inequality_op);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Set selection_flags for out-of-bounds items
      if ((IS_LAST_TILE) && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
      {
        selection_flags[ITEM] = 1;
      }
    }

    __syncthreads();

    OffsetT num_tile_selections   = 0;
    OffsetT num_selections        = 0;
    OffsetT num_selections_prefix = 0;

    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(selection_flags, selection_idx, num_tile_selections);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if constexpr (!IS_LAST_TILE)
      {
        tile_state.SetInclusive(0, num_tile_selections);
      }
    }

    // Do not count any out-of-bounds selections
    if constexpr (IS_LAST_TILE)
    {
      int num_discount = ITEMS_PER_TILE - num_tile_items;
      num_tile_selections -= num_discount;
    }
    num_selections = num_tile_selections;

    __syncthreads();

    Scatter(KeyTagT(),
            d_keys_out,
            keys,
            selection_flags,
            selection_idx,
            num_tile_items,
            num_tile_selections,
            num_selections_prefix,
            num_selections);

    __syncthreads();

    Scatter(ValueTagT(),
            d_values_out,
            values,
            selection_flags,
            selection_idx,
            num_tile_items,
            num_tile_selections,
            num_selections_prefix,
            num_selections);

    return num_selections;
  }

  /**
   * @brief Process subsequent tile of input (dynamic chained scan).
   *
   * @param num_tile_items
   *   Number of input items comprising this tile
   *
   * @param tile_idx
   *   Tile index
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @return Returns the running count of selections (including this tile)
   */
  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
  ConsumeSubsequentTile(int num_tile_items, int tile_idx, OffsetT tile_offset, ScanTileStateT& tile_state)
  {
    KeyT keys[ITEMS_PER_THREAD];
    OffsetT selection_flags[ITEMS_PER_THREAD];
    OffsetT selection_idx[ITEMS_PER_THREAD];

    if constexpr (IS_LAST_TILE)
    {
      // Fill last elements with the first element
      // because collectives are not suffix guarded
      BlockLoadKeys(temp_storage.load_keys)
        .Load(d_keys_in + tile_offset, keys, num_tile_items, *(d_keys_in + tile_offset));
    }
    else
    {
      BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);
    }

    __syncthreads();

    ValueT values[ITEMS_PER_THREAD];
    if constexpr (IS_LAST_TILE)
    {
      // Fill last elements with the first element
      // because collectives are not suffix guarded
      BlockLoadValues(temp_storage.load_values)
        .Load(d_values_in + tile_offset, values, num_tile_items, *(d_values_in + tile_offset));
    }
    else
    {
      BlockLoadValues(temp_storage.load_values).Load(d_values_in + tile_offset, values);
    }

    __syncthreads();

    KeyT tile_predecessor = d_keys_in[tile_offset - 1];
    BlockDiscontinuityKeys(temp_storage.scan_storage.discontinuity)
      .FlagHeads(selection_flags, keys, inequality_op, tile_predecessor);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Set selection_flags for out-of-bounds items
      if ((IS_LAST_TILE) && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
      {
        selection_flags[ITEM] = 1;
      }
    }

    __syncthreads();

    OffsetT num_tile_selections   = 0;
    OffsetT num_selections        = 0;
    OffsetT num_selections_prefix = 0;

    TilePrefixCallback prefix_cb(tile_state, temp_storage.scan_storage.prefix, ::cuda::std::plus<>{}, tile_idx);
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(selection_flags, selection_idx, prefix_cb);

    num_selections        = prefix_cb.GetInclusivePrefix();
    num_tile_selections   = prefix_cb.GetBlockAggregate();
    num_selections_prefix = prefix_cb.GetExclusivePrefix();

    if constexpr (IS_LAST_TILE)
    {
      int num_discount = ITEMS_PER_TILE - num_tile_items;
      num_tile_selections -= num_discount;
      num_selections -= num_discount;
    }

    __syncthreads();

    Scatter(KeyTagT(),
            d_keys_out,
            keys,
            selection_flags,
            selection_idx,
            num_tile_items,
            num_tile_selections,
            num_selections_prefix,
            num_selections);

    __syncthreads();

    Scatter(ValueTagT(),
            d_values_out,
            values,
            selection_flags,
            selection_idx,
            num_tile_items,
            num_tile_selections,
            num_selections_prefix,
            num_selections);

    return num_selections;
  }

  /**
   * @brief Process a tile of input
   *
   * @param num_tile_items
   *   Number of input items comprising this tile
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
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
  ConsumeTile(int num_tile_items, int tile_idx, OffsetT tile_offset, ScanTileStateT& tile_state)
  {
    OffsetT num_selections;
    if (tile_idx == 0)
    {
      num_selections = ConsumeFirstTile<IS_LAST_TILE>(num_tile_items, tile_offset, tile_state);
    }
    else
    {
      num_selections = ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items, tile_idx, tile_offset, tile_state);
    }

    return num_selections;
  }

  /**
   * @brief Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_tiles
   *   Total number of input tiles
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @param d_num_selected_out
   *   Output total number selection_flags
   *
   * @tparam NumSelectedIteratorT
   *   Output iterator type for recording number of items selection_flags
   *
   */
  template <typename NumSelectedIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeRange(int num_tiles, ScanTileStateT& tile_state, NumSelectedIteratorT d_num_selected_out)
  {
    // Blocks are launched in increasing order, so just assign one tile per block
    int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y; // Current tile index

    // Global offset for the current tile
    OffsetT tile_offset = static_cast<OffsetT>(tile_idx) * static_cast<OffsetT>(ITEMS_PER_TILE);

    if (tile_idx < num_tiles - 1)
    {
      ConsumeTile<false>(ITEMS_PER_TILE, tile_idx, tile_offset, tile_state);
    }
    else
    {
      int num_remaining      = static_cast<int>(num_items - tile_offset);
      OffsetT num_selections = ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
      if (threadIdx.x == 0)
      {
        *d_num_selected_out = num_selections;
      }
    }
  }
};
} // namespace detail::unique_by_key

CUB_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/util.h>

CUB_NAMESPACE_BEGIN

template <int BlockThreads,
          int ItemsPerThread                      = 1,
          cub::BlockLoadAlgorithm LoadAlgorithm   = cub::BLOCK_LOAD_DIRECT,
          cub::CacheLoadModifier LoadModifier     = cub::LOAD_LDG,
          cub::BlockStoreAlgorithm StoreAlgorithm = cub::BLOCK_STORE_DIRECT>
struct AgentAdjacentDifferencePolicy
{
  static constexpr int BLOCK_THREADS    = BlockThreads;
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = LoadAlgorithm;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = LoadModifier;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = StoreAlgorithm;
};

namespace detail::adjacent_difference
{
template <typename Policy,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT,
          typename OffsetT,
          typename InputT,
          typename OutputT,
          bool MayAlias,
          bool ReadLeft>
struct AgentDifference
{
  using LoadIt = try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, InputIteratorT>;

  using BlockLoad  = typename cub::BlockLoadType<Policy, LoadIt>::type;
  using BlockStore = typename cub::BlockStoreType<Policy, OutputIteratorT, OutputT>::type;

  using BlockAdjacentDifferenceT = cub::BlockAdjacentDifference<InputT, Policy::BLOCK_THREADS>;

  union _TempStorage
  {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
    typename BlockAdjacentDifferenceT::TempStorage adjacent_difference;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  static constexpr int BLOCK_THREADS      = Policy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD   = Policy::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE     = Policy::ITEMS_PER_TILE;
  static constexpr int SHARED_MEMORY_SIZE = static_cast<int>(sizeof(TempStorage));

  _TempStorage& temp_storage;
  InputIteratorT input_it;
  LoadIt load_it;
  InputT* first_tile_previous;
  OutputIteratorT result;
  DifferenceOpT difference_op;
  OffsetT num_items;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentDifference(
    TempStorage& temp_storage,
    InputIteratorT input_it,
    InputT* first_tile_previous,
    OutputIteratorT result,
    DifferenceOpT difference_op,
    OffsetT num_items)
      : temp_storage(temp_storage.Alias())
      , input_it(input_it)
      , load_it(LoadIt(input_it))
      , first_tile_previous(first_tile_previous)
      , result(result)
      , difference_op(difference_op)
      , num_items(num_items)
  {}

  template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile_impl(int num_remaining, int tile_idx, OffsetT tile_base)
  {
    InputT input[ITEMS_PER_THREAD];
    OutputT output[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last elements with the first element
      // because collectives are not suffix guarded
      BlockLoad(temp_storage.load).Load(load_it + tile_base, input, num_remaining, *(load_it + tile_base));
    }
    else
    {
      BlockLoad(temp_storage.load).Load(load_it + tile_base, input);
    }

    __syncthreads();

    if (ReadLeft)
    {
      if (IS_FIRST_TILE)
      {
        if (IS_LAST_TILE)
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeftPartialTile(input, output, difference_op, num_remaining);
        }
        else
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference).SubtractLeft(input, output, difference_op);
        }
      }
      else
      {
        InputT tile_prev_input = MayAlias ? first_tile_previous[tile_idx] : *(input_it + tile_base - 1);

        if (IS_LAST_TILE)
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeftPartialTile(input, output, difference_op, num_remaining, tile_prev_input);
        }
        else
        {
          BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
            .SubtractLeft(input, output, difference_op, tile_prev_input);
        }
      }
    }
    else
    {
      if (IS_LAST_TILE)
      {
        BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
          .SubtractRightPartialTile(input, output, difference_op, num_remaining);
      }
      else
      {
        InputT tile_next_input = MayAlias ? first_tile_previous[tile_idx] : *(input_it + tile_base + ITEMS_PER_TILE);

        BlockAdjacentDifferenceT(temp_storage.adjacent_difference)
          .SubtractRight(input, output, difference_op, tile_next_input);
      }
    }

    __syncthreads();

    if (IS_LAST_TILE)
    {
      BlockStore(temp_storage.store).Store(result + tile_base, output, num_remaining);
    }
    else
    {
      BlockStore(temp_storage.store).Store(result + tile_base, output);
    }
  }

  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(int num_remaining, int tile_idx, OffsetT tile_base)
  {
    if (tile_idx == 0)
    {
      consume_tile_impl<IS_LAST_TILE, true>(num_remaining, tile_idx, tile_base);
    }
    else
    {
      consume_tile_impl<IS_LAST_TILE, false>(num_remaining, tile_idx, tile_base);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process(int tile_idx, OffsetT tile_base)
  {
    OffsetT num_remaining = num_items - tile_base;

    if (num_remaining > ITEMS_PER_TILE) // not a last tile
    {
      consume_tile<false>(num_remaining, tile_idx, tile_base);
    }
    else
    {
      consume_tile<true>(num_remaining, tile_idx, tile_base);
    }
  }
};

template <typename InputIteratorT, typename InputT, typename OffsetT, bool ReadLeft>
struct AgentDifferenceInit
{
  static constexpr int BLOCK_THREADS = 128;

  static _CCCL_DEVICE _CCCL_FORCEINLINE void
  Process(int tile_idx, InputIteratorT first, InputT* result, OffsetT num_tiles, int items_per_tile)
  {
    OffsetT tile_base = static_cast<OffsetT>(tile_idx) * items_per_tile;

    if (tile_base > 0 && tile_idx < num_tiles)
    {
      if (ReadLeft)
      {
        result[tile_idx] = first[tile_base - 1];
      }
      else
      {
        result[tile_idx - 1] = first[tile_base];
      }
    }
  }
};
} // namespace detail::adjacent_difference

CUB_NAMESPACE_END

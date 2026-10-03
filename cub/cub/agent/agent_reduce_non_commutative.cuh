// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Thread block abstraction for a device-wide reduction that combines items in input order.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_pointer.h>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_non_commutative
{
//! Splits [0, num_items) into contiguous runs of whole tiles, one run per share, in input order. The first
//! `big_shares` shares take one tile more than the others. Unlike GridEvenShare, the tile count is not limited to int.
template <typename OffsetT>
struct even_share_t
{
  OffsetT num_items;
  OffsetT tiles_per_share;
  OffsetT big_shares;
  int num_shares;

  //! @pre num_items > 0
  [[nodiscard]] _CCCL_HOST_DEVICE_API static even_share_t make(OffsetT num_items, int max_shares, int tile_items)
  {
    const OffsetT total_tiles = ::cuda::ceil_div(num_items, static_cast<OffsetT>(tile_items));
    const int num_shares      = static_cast<int>(::cuda::std::min(total_tiles, static_cast<OffsetT>(max_shares)));
    return {num_items,
            total_tiles / static_cast<OffsetT>(num_shares),
            total_tiles % static_cast<OffsetT>(num_shares),
            num_shares};
  }

  //! Writes the bounds of share `share`, which must be less than `num_shares`.
  template <int TileItems>
  _CCCL_DEVICE_API void range(int share, OffsetT& begin, OffsetT& end) const
  {
    const auto share_offset  = static_cast<OffsetT>(share);
    const OffsetT first_tile = share_offset * tiles_per_share + ::cuda::std::min(share_offset, big_shares);
    const OffsetT num_tiles  = tiles_per_share + (share_offset < big_shares ? 1 : 0);

    begin = first_tile * TileItems;

    // The last share may end in a partial tile, and `num_tiles * TileItems` may not fit in OffsetT there.
    const OffsetT remaining = num_items - begin;
    end = num_tiles >= ::cuda::ceil_div(remaining, OffsetT{TileItems}) ? num_items : begin + num_tiles * TileItems;
  }
};

//! Reduces a range so that every application of the reduction operator combines two neighboring runs of the input,
//! the earlier one as the left operand. The operator therefore needs to be associative but not commutative.
//!
//! Each warp owns a contiguous run of warp tiles. Within a tile, each lane folds its consecutive items in order, the
//! warp reduction combines the lanes in lane order, and lane 0 folds the tile aggregates in order. The block then
//! folds the warp aggregates in warp order.
template <int ThreadsPerBlock,
          int ItemsPerThread,
          CacheLoadModifier LoadModifier,
          typename InputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
struct agent_t
{
  static_assert(ThreadsPerBlock % warp_threads == 0,
                "The number of threads per block must be a multiple of the warp size");

  static constexpr int warps_per_block = ThreadsPerBlock / warp_threads;
  static constexpr int warp_tile_items = warp_threads * ItemsPerThread;

  using input_t       = it_value_t<InputIteratorT>;
  using warp_reduce_t = WarpReduce<AccumT>;

  static constexpr bool attempt_vectorization =
    ::cuda::std::is_pointer_v<InputIteratorT> && ::cuda::is_trivially_copyable_v<input_t>;

  struct _TempStorage
  {
    typename warp_reduce_t::TempStorage warp_reduce[warps_per_block];
    AccumT warp_aggregates[warps_per_block];
  };

  using TempStorage = Uninitialized<_TempStorage>;

  _TempStorage& temp_storage;
  InputIteratorT d_in;
  ReductionOpT reduction_op;
  TransformOpT transform_op;
  int warp_id;
  int lane_id;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE
  agent_t(TempStorage& temp_storage, InputIteratorT d_in, ReductionOpT reduction_op, TransformOpT transform_op)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , reduction_op(reduction_op)
      , transform_op(transform_op)
      , warp_id(static_cast<int>(threadIdx.x) / warp_threads)
      , lane_id(static_cast<int>(threadIdx.x) % warp_threads)
  {}

  //! Reduces the full warp tile at `tile_offset`. The result is valid in lane 0.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT consume_full_warp_tile(OffsetT tile_offset)
  {
    input_t items[ItemsPerThread];
    if constexpr (attempt_vectorization)
    {
      // Falls back to scalar loads if the tile is not aligned for vector loads
      InternalLoadDirectBlockedVectorized<LoadModifier>(lane_id, d_in + tile_offset, items);
    }
    else
    {
      LoadDirectBlocked(lane_id, d_in + tile_offset, items);
    }

    AccumT thread_aggregate = transform_op(items[0]);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 1; i < ItemsPerThread; ++i)
    {
      const AccumT item = transform_op(items[i]);
      thread_aggregate  = reduction_op(thread_aggregate, item);
    }
    return warp_reduce_t(temp_storage.warp_reduce[warp_id]).Reduce(thread_aggregate, reduction_op);
  }

  //! Reduces the `num_valid` items at `tile_offset`, fewer than a full warp tile. The result is valid in lane 0.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT consume_partial_warp_tile(OffsetT tile_offset, int num_valid)
  {
    input_t items[ItemsPerThread];
    LoadDirectBlocked(lane_id, d_in + tile_offset, items, num_valid);

    const int thread_items  = num_valid - lane_id * ItemsPerThread;
    AccumT thread_aggregate = {};
    if (thread_items > 0)
    {
      thread_aggregate = transform_op(items[0]);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < ItemsPerThread; ++i)
      {
        if (i < thread_items)
        {
          const AccumT item = transform_op(items[i]);
          thread_aggregate  = reduction_op(thread_aggregate, item);
        }
      }
    }
    const int valid_lanes = ::cuda::ceil_div(num_valid, ItemsPerThread);
    return warp_reduce_t(temp_storage.warp_reduce[warp_id]).Reduce(thread_aggregate, reduction_op, valid_lanes);
  }

  //! Reduces the non-empty range [begin, end) with the calling warp. The result is valid in lane 0.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT consume_warp_range(OffsetT begin, OffsetT end)
  {
    constexpr auto tile_items = static_cast<OffsetT>(warp_tile_items);

    OffsetT remaining = end - begin;
    if (remaining < tile_items)
    {
      return consume_partial_warp_tile(begin, static_cast<int>(remaining));
    }

    AccumT warp_aggregate = consume_full_warp_tile(begin);
    begin += tile_items;
    remaining -= tile_items;

    while (remaining >= tile_items)
    {
      const AccumT tile_aggregate = consume_full_warp_tile(begin);
      if (lane_id == 0)
      {
        warp_aggregate = reduction_op(warp_aggregate, tile_aggregate);
      }
      begin += tile_items;
      remaining -= tile_items;
    }

    if (remaining > 0)
    {
      const AccumT tile_aggregate = consume_partial_warp_tile(begin, static_cast<int>(remaining));
      if (lane_id == 0)
      {
        warp_aggregate = reduction_op(warp_aggregate, tile_aggregate);
      }
    }
    return warp_aggregate;
  }

  //! Reduces the shares `first_share + warp_id` of `even_share`, one per warp, and folds the warp aggregates in warp
  //! order. `first_share` must be less than `even_share.num_shares`. The result is valid in thread 0.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT
  consume_shares(const even_share_t<OffsetT>& even_share, int first_share)
  {
    const int block_shares = ::cuda::std::min(int{warps_per_block}, even_share.num_shares - first_share);

    if (warp_id < block_shares)
    {
      OffsetT begin;
      OffsetT end;
      even_share.template range<warp_tile_items>(first_share + warp_id, begin, end);
      const AccumT warp_aggregate = consume_warp_range(begin, end);
      if (lane_id == 0)
      {
        temp_storage.warp_aggregates[warp_id] = warp_aggregate;
      }
    }
    __syncthreads();

    AccumT block_aggregate = {};
    if (threadIdx.x == 0)
    {
      block_aggregate = temp_storage.warp_aggregates[0];
      for (int warp = 1; warp < block_shares; ++warp)
      {
        block_aggregate = reduction_op(block_aggregate, temp_storage.warp_aggregates[warp]);
      }
    }
    return block_aggregate;
  }

  //! Reduces the non-empty range [0, num_items) with this block alone. The result is valid in thread 0.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE AccumT consume_range(OffsetT num_items)
  {
    return consume_shares(even_share_t<OffsetT>::make(num_items, warps_per_block, warp_tile_items), 0);
  }
};
} // namespace detail::reduce_non_commutative

CUB_NAMESPACE_END

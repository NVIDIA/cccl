// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * The cub::WarpExchangeSmem class provides [<em>collective</em>](index.html#sec0)
 * methods for rearranging data partitioned across a CUDA warp.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__ptx/instructions/get_sreg.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename InputT, int ItemsPerThread, int LogicalWarpThreads = warp_threads>
class WarpExchangeSmem
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");

  static constexpr int ITEMS_PER_TILE = ItemsPerThread * LogicalWarpThreads + 1;

  static constexpr bool IS_ARCH_WARP = LogicalWarpThreads == warp_threads;

  static constexpr int LOG_SMEM_BANKS = log2_smem_banks;

  // Insert padding if the number of items per thread is a power of two
  // and > 4 (otherwise we can typically use 128b loads)
  static constexpr bool INSERT_PADDING = (ItemsPerThread > 4) && (::cuda::is_power_of_two(ItemsPerThread));

  static constexpr int PADDING_ITEMS = INSERT_PADDING ? (ITEMS_PER_TILE >> LOG_SMEM_BANKS) : 0;

  union _TempStorage
  {
    InputT items_shared[ITEMS_PER_TILE + PADDING_ITEMS];
  }; // union TempStorage

  /// Shared storage reference
  _TempStorage& temp_storage;

  const unsigned int lane_id;
  const unsigned int warp_id;
  const unsigned int member_mask;

public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  WarpExchangeSmem() = delete;

  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpExchangeSmem(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LogicalWarpThreads))
      , warp_id(IS_ARCH_WARP ? 0 : (::cuda::ptx::get_sreg_laneid() / LogicalWarpThreads))
      , member_mask(WarpMask<LogicalWarpThreads>(warp_id))
  {}

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  BlockedToStriped(const InputT (&input_items)[ItemsPerThread], OutputT (&output_items)[ItemsPerThread])
  {
    for (int item = 0; item < ItemsPerThread; item++)
    {
      const int idx                  = ItemsPerThread * lane_id + item;
      temp_storage.items_shared[idx] = input_items[item];
    }
    __syncwarp(member_mask);

    for (int item = 0; item < ItemsPerThread; item++)
    {
      const int idx      = LogicalWarpThreads * item + lane_id;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StripedToBlocked(const InputT (&input_items)[ItemsPerThread], OutputT (&output_items)[ItemsPerThread])
  {
    for (int item = 0; item < ItemsPerThread; item++)
    {
      const int idx                  = LogicalWarpThreads * item + lane_id;
      temp_storage.items_shared[idx] = input_items[item];
    }
    __syncwarp(member_mask);

    for (int item = 0; item < ItemsPerThread; item++)
    {
      const int idx      = ItemsPerThread * lane_id + item;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(InputT (&items)[ItemsPerThread], OffsetT (&ranks)[ItemsPerThread])
  {
    ScatterToStriped(items, items, ranks);
  }

  template <typename OutputT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(
    const InputT (&input_items)[ItemsPerThread],
    OutputT (&output_items)[ItemsPerThread],
    OffsetT (&ranks)[ItemsPerThread])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
    {
      if (INSERT_PADDING)
      {
        ranks[ITEM] = (ranks[ITEM] >> LOG_SMEM_BANKS) + ranks[ITEM];
      }

      temp_storage.items_shared[ranks[ITEM]] = input_items[ITEM];
    }

    __syncwarp(member_mask);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
    {
      int item_offset = (ITEM * LogicalWarpThreads) + lane_id;

      if (INSERT_PADDING)
      {
        item_offset = (item_offset >> LOG_SMEM_BANKS) + item_offset;
      }

      output_items[ITEM] = temp_storage.items_shared[item_offset];
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END

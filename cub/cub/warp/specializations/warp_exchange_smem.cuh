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
template <typename InputT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS = warp_threads>
class WarpExchangeSmem
{
  static_assert(::cuda::is_power_of_two(LOGICAL_WARP_THREADS), "LOGICAL_WARP_THREADS must be a power of two");

  static constexpr int ITEMS_PER_TILE = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS + 1;

  static constexpr bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == warp_threads;

  static constexpr int LOG_SMEM_BANKS = log2_smem_banks;

  // Insert padding if the number of items per thread is a power of two
  // and > 4 (otherwise we can typically use 128b loads)
  static constexpr bool INSERT_PADDING = (ITEMS_PER_THREAD > 4) && (::cuda::is_power_of_two(ITEMS_PER_THREAD));

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
      , lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {}

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  BlockedToStriped(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx                  = ITEMS_PER_THREAD * lane_id + item;
      temp_storage.items_shared[idx] = input_items[item];
    }
    __syncwarp(member_mask);

    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx      = LOGICAL_WARP_THREADS * item + lane_id;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StripedToBlocked(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx                  = LOGICAL_WARP_THREADS * item + lane_id;
      temp_storage.items_shared[idx] = input_items[item];
    }
    __syncwarp(member_mask);

    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx      = ITEMS_PER_THREAD * lane_id + item;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScatterToStriped(InputT (&items)[ITEMS_PER_THREAD], OffsetT (&ranks)[ITEMS_PER_THREAD])
  {
    ScatterToStriped(items, items, ranks);
  }

  template <typename OutputT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(
    const InputT (&input_items)[ITEMS_PER_THREAD],
    OutputT (&output_items)[ITEMS_PER_THREAD],
    OffsetT (&ranks)[ITEMS_PER_THREAD])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      if (INSERT_PADDING)
      {
        ranks[ITEM] = (ranks[ITEM] >> LOG_SMEM_BANKS) + ranks[ITEM];
      }

      temp_storage.items_shared[ranks[ITEM]] = input_items[ITEM];
    }

    __syncwarp(member_mask);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      int item_offset = (ITEM * LOGICAL_WARP_THREADS) + lane_id;

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

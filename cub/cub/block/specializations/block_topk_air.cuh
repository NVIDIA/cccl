// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_scan.cuh>
#include <cub/block/block_topk_rank.cuh>
#include <cub/block/specializations/block_topk_rank_atomic.cuh>
#include <cub/block/specializations/block_topk_sieve_air.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @brief Block-level top-k by radix selection.
//!
//! Selects the smallest (or largest) @p k keys from a tile of keys in registers, without
//! fully sorting. The algorithm has two stages: (1) Radix selection determines the bit-prefix
//! of the k-th key by processing bits MSB to LSB in passes of @p RadixBits. In each pass, a
//! histogram over the current digit is built over candidates only (keys matching the prefix so
//! far), then a prefix sum identifies the bucket containing the k-th item. Items in earlier
//! buckets are guaranteed top-k; items in later buckets are discarded; the chosen bucket
//! becomes the candidate set for the next pass. No data movement occurs during this stage—only
//! the histogram in shared memory is updated. (2) Partitioning scatters the top-k items (key
//! prefix <= k-th prefix) into shared memory via atomic counters, then each thread reads back
//! its portion. Supports key-only and key-value selection.
template <typename KeyT, int ThreadsPerBlock, int ItemsPerThread, typename ValueT = NullType>
class block_topk_air
{
private:
  // TODO (elstehle): Make this configurable
  // Whether to include all items tied with the k-th key when selecting top-k
  static constexpr bool expand_k_to_include_ties = false;

  static constexpr int threads_per_block = ThreadsPerBlock;
  static constexpr int items_per_thread  = ItemsPerThread;
  static constexpr int tile_items        = threads_per_block * items_per_thread;

  // Calculate number of buckets processed per thread
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;

  using block_sieve_t         = block_topk_sieve<KeyT, threads_per_block>;
  using block_sieve_storage_t = typename block_sieve_t::TempStorage;
  using block_rank_t          = block_topk_rank<threads_per_block>;
  using block_rank_storage_t  = typename block_rank_t::TempStorage;

  static_assert(
    ::cuda::std::is_base_of_v<Uninitialized<typename block_topk_sieve_air<KeyT, threads_per_block>::TempStorage>,
                              block_sieve_storage_t>,
    "Wrong sieve specialization");
  static_assert(::cuda::std::is_base_of_v<Uninitialized<typename block_topk_rank_atomic<threads_per_block>::TempStorage>,
                                          block_rank_storage_t>,
                "Wrong rank specialization");

  struct TempStorage_
  {
    union
    {
      block_sieve_storage_t sieve_storage;

      struct
      {
        block_rank_storage_t rank_storage;
        union
        {
          KeyT keys[tile_items];
          ValueT values[tile_items];
        } exchange;
      } select;
    } stage;
  };

  /// Shared storage reference
  TempStorage_& storage;

  /// Linear thread index
  int linear_tid;

  template <detail::topk::select Dir, bool Full>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static block_topk_key_states<items_per_thread> sieve_select(
    block_sieve_storage_t& sieve_storage,
    KeyT (&keys)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit,
    int end_bit)
  {
    if constexpr (Dir == detail::topk::select::max)
    {
      return block_sieve_t(sieve_storage).template select_max<Full>(keys, k, valid_items, begin_bit, end_bit);
    }
    else
    {
      return block_sieve_t(sieve_storage).template select_min<Full>(keys, k, valid_items, begin_bit, end_bit);
    }
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_topk(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit,
    int end_bit)
  {
    if constexpr (!IsFullTile)
    {
      _CCCL_ASSERT(valid_items > 0 && valid_items <= tile_items, "valid_items must be in [1, tile_items]");
    }

    // TODO (elstehle): Short-circuit if begin_bit is constrained to be non-negative
    begin_bit = (::cuda::std::max) (begin_bit, 0);

    // TODO (elstehle): Short-circuit if end_bit is constrained to be less than the maximum number of bits in the key
    // type
    const int max_bit = int(sizeof(KeyT) * 8);
    if (end_bit > max_bit)
    {
      end_bit = max_bit;
    }

    // TODO (elstehle): Short-circuit if k is greater than the number of items in the tile
    if ((!IsFullTile && k >= valid_items) || k >= tile_items)
    {
      return;
    }

    auto states =
      sieve_select<SelectDirection, IsFullTile>(storage.stage.sieve_storage, keys, k, valid_items, begin_bit, end_bit);
    // Make sure smem can be reused by the rank stage
    __syncthreads();

    int scatter_indices[items_per_thread];
    block_rank_t(storage.stage.select.rank_storage).rank_key_states(states, scatter_indices);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      if (scatter_indices[i] >= 0)
      {
        storage.stage.select.exchange.keys[scatter_indices[i]] = keys[i];
      }
    }

    // Ensure all threads have finished writing to shared memory
    __syncthreads();

    // Gather selected items into thread registers for return.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const int buffer_idx = linear_tid * items_per_thread + i;
      if (buffer_idx < k)
      {
        keys[i] = storage.stage.select.exchange.keys[buffer_idx];
      }
    }

    if constexpr (!keys_only)
    {
      // Ensure all keys have been loaded from shared memory before we repurpose the exchange buffer for values
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        if (scatter_indices[i] >= 0)
        {
          storage.stage.select.exchange.values[scatter_indices[i]] = values[i];
        }
      }

      // Ensure all values have been written to shared memory before we read them back in
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        const int buffer_idx = linear_tid * items_per_thread + i;
        if (buffer_idx < k)
        {
          values[i] = storage.stage.select.exchange.values[buffer_idx];
        }
      }
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE block_topk_air(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(ThreadsPerBlock, 1, 1))
  {}

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  select_keys(KeyT (&keys)[items_per_thread], int k, int valid_items, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    NullType values[ItemsPerThread];
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items, begin_bit, end_bit);
  }

  template <detail::topk::select SelectDirection, bool IsFullTile>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void select_pairs(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int valid_items,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    select_topk<SelectDirection, IsFullTile>(keys, values, k, valid_items, begin_bit, end_bit);
  }
};
} // namespace detail
CUB_NAMESPACE_END

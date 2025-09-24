// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN
namespace detail::merge
{
template <int ThreadsPerBlock, int ItemsPerThread, CacheLoadModifier LoadCacheModifier, BlockStoreAlgorithm StoreAlgorithm>
struct agent_policy_t
{
  // do not change data member names, policy_wrapper_t depends on it
  static constexpr int BLOCK_THREADS                   = ThreadsPerBlock;
  static constexpr int ITEMS_PER_THREAD                = ItemsPerThread;
  static constexpr int ITEMS_PER_TILE                  = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = LoadCacheModifier;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = StoreAlgorithm;
};

// TODO(bgruber): can we unify this one with AgentMerge in agent_merge_sort.cuh?
template <typename Policy,
          typename KeysIt1,
          typename ItemsIt1,
          typename KeysIt2,
          typename ItemsIt2,
          typename KeysOutputIt,
          typename ItemsOutputIt,
          typename Offset,
          typename CompareOp>
struct agent_t
{
  using policy = Policy;

  // key and value type are taken from the first input sequence (consistent with old Thrust behavior)
  using key_type  = it_value_t<KeysIt1>;
  using item_type = it_value_t<ItemsIt1>;

  using keys_load_it1  = try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, KeysIt1>;
  using keys_load_it2  = try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, KeysIt2>;
  using items_load_it1 = try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, ItemsIt1>;
  using items_load_it2 = try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, ItemsIt2>;

  using block_store_keys  = typename BlockStoreType<Policy, KeysOutputIt, key_type>::type;
  using block_store_items = typename BlockStoreType<Policy, ItemsOutputIt, item_type>::type;

  static constexpr int items_per_thread  = Policy::ITEMS_PER_THREAD;
  static constexpr int threads_per_block = Policy::BLOCK_THREADS;
  static constexpr int items_per_tile    = Policy::ITEMS_PER_TILE;

  union temp_storages
  {
    typename block_store_keys::TempStorage store_keys;
    typename block_store_items::TempStorage store_items;

    // We could change SerialMerge to avoid reading one item out of bounds and drop the + 1 here. But that would
    // introduce more branches (about 10% slower on 2^16 problem sizes on RTX 5090 in a first attempt)
    key_type keys_shared[items_per_tile + 1];
    item_type items_shared[items_per_tile + 1];
  };

  struct TempStorage : Uninitialized<temp_storages>
  {};

  // Per thread data
  temp_storages& storage;
  keys_load_it1 keys1_in;
  items_load_it1 items1_in;
  Offset keys1_count;
  keys_load_it2 keys2_in;
  items_load_it2 items2_in;
  Offset keys2_count;
  KeysOutputIt keys_out;
  ItemsOutputIt items_out;
  CompareOp compare_op;
  Offset* key1_beg_offsets;

  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(Offset tile_idx, Offset tile_base, int num_remaining)
  {
    const Offset diag0 = items_per_tile * tile_idx;
    Offset diag1       = diag0 + items_per_tile;
    if constexpr (IsFullTile)
    {
      _CCCL_ASSERT(diag1 <= keys1_count + keys2_count, "");
    }
    else
    {
      diag1 = keys1_count + keys2_count;
    }

    // compute bounding box for keys1 & keys2
    const Offset keys1_beg = key1_beg_offsets[tile_idx + 0];
    const Offset keys1_end = key1_beg_offsets[tile_idx + 1];
    const Offset keys2_beg = diag0 - keys1_beg;
    const Offset keys2_end = diag1 - keys1_end;

    // number of keys per tile
    const int keys1_count_tile = static_cast<int>(keys1_end - keys1_beg);
    const int keys2_count_tile = static_cast<int>(keys2_end - keys2_beg);
    if constexpr (IsFullTile)
    {
      _CCCL_ASSERT(keys1_count_tile + keys2_count_tile == items_per_tile, "");
    }
    else
    {
      _CCCL_ASSERT(keys1_count_tile + keys2_count_tile == num_remaining, "");
    }

    key_type keys_loc[items_per_thread];
    merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
      keys_loc, keys1_in + keys1_beg, keys2_in + keys2_beg, keys1_count_tile, keys2_count_tile);
    merge_sort::reg_to_shared<threads_per_block>(&storage.keys_shared[0], keys_loc);
    __syncthreads();

    // now find the merge path for each of thread.
    // we can use int type here, because the number of items in shared memory is limited
    int diag0_thread = items_per_thread * static_cast<int>(threadIdx.x);
    if constexpr (IsFullTile)
    {
      _CCCL_ASSERT(num_remaining == items_per_tile, "");
      _CCCL_ASSERT(diag0_thread < num_remaining, "");
    }
    else
    { // for partial tiles, clamp the thread diagonal to the valid items
      diag0_thread = (::cuda::std::min) (diag0_thread, num_remaining);
    }

    const int keys1_beg_thread = MergePath(
      &storage.keys_shared[0],
      &storage.keys_shared[keys1_count_tile],
      keys1_count_tile,
      keys2_count_tile,
      diag0_thread,
      compare_op);
    const int keys2_beg_thread = diag0_thread - keys1_beg_thread;

    const int keys1_count_thread = keys1_count_tile - keys1_beg_thread;
    const int keys2_count_thread = keys2_count_tile - keys2_beg_thread;

    // perform serial merge
    int indices[items_per_thread];
    SerialMerge(
      &storage.keys_shared[0],
      keys1_beg_thread,
      keys2_beg_thread + keys1_count_tile,
      keys1_count_thread,
      keys2_count_thread,
      keys_loc,
      indices,
      compare_op);

    // write keys
    __syncthreads(); // sync after reading from SMEM before so block store can use SMEM again
    if constexpr (IsFullTile)
    {
      block_store_keys{storage.store_keys}.Store(keys_out + tile_base, keys_loc);
    }
    else
    {
      block_store_keys{storage.store_keys}.Store(keys_out + tile_base, keys_loc, num_remaining);
    }

    // if items are provided, merge them
    static constexpr bool have_items = !::cuda::std::is_same_v<item_type, NullType>;
    if constexpr (have_items)
    {
      item_type items_loc[items_per_thread];
      merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
        items_loc, items1_in + keys1_beg, items2_in + keys2_beg, keys1_count_tile, keys2_count_tile);
      __syncthreads(); // block_store_keys above uses SMEM, so make sure all threads are done before we write to it
      merge_sort::reg_to_shared<threads_per_block>(&storage.items_shared[0], items_loc);
      __syncthreads();

      // gather items from shared mem
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        items_loc[i] = storage.items_shared[indices[i]];
      }
      __syncthreads();

      // write from reg to gmem
      if constexpr (IsFullTile)
      {
        block_store_items{storage.store_items}.Store(items_out + tile_base, items_loc);
      }
      else
      {
        block_store_items{storage.store_items}.Store(items_out + tile_base, items_loc, num_remaining);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()()
  {
    const Offset tile_idx  = blockIdx.x;
    const Offset tile_base = tile_idx * items_per_tile;
    const int items_in_tile =
      static_cast<int>((::cuda::std::min) (static_cast<Offset>(items_per_tile), keys1_count + keys2_count - tile_base));
    if (items_in_tile == items_per_tile)
    {
      consume_tile</* IsFullTile */ true>(tile_idx, tile_base, items_per_tile);
    }
    else
    {
      consume_tile</* IsFullTile */ false>(tile_idx, tile_base, items_in_tile);
    }
  }
};
} // namespace detail::merge
CUB_NAMESPACE_END

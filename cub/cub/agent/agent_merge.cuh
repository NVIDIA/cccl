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
#include <cub/block/block_load_to_shared.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__memory/ptr_rebind.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/cstddef>
#include <cuda/std/span>

CUB_NAMESPACE_BEGIN
namespace detail::merge
{
template <int ThreadsPerBlock,
          int ItemsPerThread,
          CacheLoadModifier LoadCacheModifier,
          BlockStoreAlgorithm StoreAlgorithm,
          bool UseBlockLoadToShared = false>
struct agent_policy_t
{
  // do not change data member names, policy_wrapper_t depends on it
  static constexpr int BLOCK_THREADS                   = ThreadsPerBlock;
  static constexpr int ITEMS_PER_THREAD                = ItemsPerThread;
  static constexpr int ITEMS_PER_TILE                  = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = LoadCacheModifier;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = StoreAlgorithm;
  static constexpr bool use_block_load_to_shared       = UseBlockLoadToShared;
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
  using policy                           = Policy;
  static constexpr int items_per_thread  = Policy::ITEMS_PER_THREAD;
  static constexpr int threads_per_block = Policy::BLOCK_THREADS;
  static constexpr int items_per_tile    = Policy::ITEMS_PER_TILE;

  // key and value type are taken from the first input sequence (consistent with old Thrust behavior)
  using key_type  = it_value_t<KeysIt1>;
  using item_type = it_value_t<ItemsIt1>;

  using block_load_to_shared = cub::detail::BlockLoadToShared<threads_per_block>;
  using block_store_keys     = typename BlockStoreType<Policy, KeysOutputIt, key_type>::type;
  using block_store_items    = typename BlockStoreType<Policy, ItemsOutputIt, item_type>::type;

  template <typename ValueT, typename Iter1, typename Iter2>
  static constexpr bool use_block_load_to_shared =
    Policy::use_block_load_to_shared && (sizeof(ValueT) == alignof(ValueT))
    && THRUST_NS_QUALIFIER::is_trivially_relocatable_v<ValueT> //
    && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<Iter1> //
    && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<Iter2>
    && ::cuda::std::is_same_v<ValueT, cub::detail::it_value_t<Iter1>>
    && ::cuda::std::is_same_v<ValueT, cub::detail::it_value_t<Iter2>>;

  static constexpr bool keys_use_bl2sh     = use_block_load_to_shared<key_type, KeysIt1, KeysIt2>;
  static constexpr bool items_use_bl2sh    = use_block_load_to_shared<item_type, ItemsIt1, ItemsIt2>;
  static constexpr int bl2sh_minimum_align = block_load_to_shared::template SharedBufferAlignBytes<char>();

  template <typename ValueT>
  struct alignas(block_load_to_shared::template SharedBufferAlignBytes<ValueT>()) buffer_t
  {
    // Need extra bytes of padding for TMA because this static buffer has to hold the two dynamically sized buffers.
    static constexpr int bytes_needed = block_load_to_shared::template SharedBufferSizeBytes<ValueT>(items_per_tile + 1)
                                      + (alignof(ValueT) < bl2sh_minimum_align ? 2 * bl2sh_minimum_align : 0);

    char c_array[bytes_needed];
  };

  struct temp_storages_without_bl2sh
  {
    using keys_smem  = ::cuda::std::conditional_t<keys_use_bl2sh, buffer_t<key_type>, key_type[items_per_tile + 1]>;
    using items_smem = ::cuda::std::conditional_t<items_use_bl2sh, buffer_t<item_type>, item_type[items_per_tile + 1]>;
    union
    {
      typename block_store_keys::TempStorage store_keys;
      typename block_store_items::TempStorage store_items;
      keys_smem keys_shared;
      items_smem items_shared;
    };
  };

  // inherit from data storage, so it's positioned at the start of the shared memory
  struct temp_storages_with_bl2sh : temp_storages_without_bl2sh
  {
    typename block_load_to_shared::TempStorage load2sh;
  };

  using temp_storages =
    ::cuda::std::conditional_t<keys_use_bl2sh || items_use_bl2sh, temp_storages_with_bl2sh, temp_storages_without_bl2sh>;

  using TempStorage = Uninitialized<temp_storages>;

  // Per thread data
  temp_storages& storage;
  KeysIt1 keys1_in;
  ItemsIt1 items1_in;
  Offset keys1_count;
  KeysIt2 keys2_in;
  ItemsIt2 items2_in;
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

    [[maybe_unused]] auto load2sh = [&] {
      if constexpr (keys_use_bl2sh || items_use_bl2sh)
      {
        return block_load_to_shared{storage.load2sh};
      }
      else
      {
        return NullType{};
      }
    }();

    key_type keys_loc[items_per_thread];
    key_type* keys1_shared;
    key_type* keys2_shared;
    int keys2_offset;
    if constexpr (keys_use_bl2sh)
    {
      ::cuda::std::span keys1_src{THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(keys1_in + keys1_beg),
                                  static_cast<::cuda::std::size_t>(keys1_count_tile)};
      ::cuda::std::span keys2_src{THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(keys2_in + keys2_beg),
                                  static_cast<::cuda::std::size_t>(keys2_count_tile)};
      ::cuda::std::span keys_buffers{storage.keys_shared.c_array};
      auto keys1_buffer =
        keys_buffers.first(block_load_to_shared::template SharedBufferSizeBytes<key_type>(keys1_count_tile));
      auto keys2_buffer =
        keys_buffers.last(block_load_to_shared::template SharedBufferSizeBytes<key_type>(keys2_count_tile));
      _CCCL_ASSERT(keys1_buffer.end() <= keys2_buffer.begin(),
                   "Keys buffer needs to be appropriately sized (internal)");
      keys1_shared = data(load2sh.CopyAsync(keys1_buffer, keys1_src));
      keys2_shared = data(load2sh.CopyAsync(keys2_buffer, keys2_src));
      auto token   = load2sh.Commit();
      // Needed for using keys1_shared as one big buffer including both ranges in SerialMerge
      keys2_offset = static_cast<int>(keys2_shared - keys1_shared);
      load2sh.Wait(::cuda::std::move(token));
    }
    else
    {
      auto keys1_in_cm = try_make_cache_modified_iterator<Policy::LOAD_MODIFIER>(keys1_in);
      auto keys2_in_cm = try_make_cache_modified_iterator<Policy::LOAD_MODIFIER>(keys2_in);
      merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
        keys_loc, keys1_in_cm + keys1_beg, keys2_in_cm + keys2_beg, keys1_count_tile, keys2_count_tile);
      keys1_shared = &storage.keys_shared[0];
      // Needed for using keys1_shared as one big buffer including both ranges in SerialMerge
      keys2_offset = keys1_count_tile;
      keys2_shared = keys1_shared + keys2_offset;
      merge_sort::reg_to_shared<threads_per_block>(keys1_shared, keys_loc);
      __syncthreads();
    }

    // Now find the merge path for each of the threads.
    // We can use int type here, because the number of items in shared memory is limited.
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

    const int keys1_beg_thread =
      MergePath(keys1_shared, keys2_shared, keys1_count_tile, keys2_count_tile, diag0_thread, compare_op);
    const int keys2_beg_thread = diag0_thread - keys1_beg_thread;

    const int keys1_count_thread = keys1_count_tile - keys1_beg_thread;
    const int keys2_count_thread = keys2_count_tile - keys2_beg_thread;

    // perform serial merge
    int indices[items_per_thread];
    cub::SerialMerge(
      keys1_shared,
      keys1_beg_thread,
      keys2_offset + keys2_beg_thread,
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
      // Both of these are only needed when either keys or items or both use BlockLoadToShared introducing padding (that
      // can differ between the keys and items)
      [[maybe_unsused]] const auto translate_indices = [&](int items2_offset) -> void {
        const int diff = items2_offset - keys2_offset;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < items_per_thread; ++i)
        {
          if (indices[i] >= keys2_offset)
          {
            indices[i] += diff;
          }
        }
      };
      // WAR for MSVC erroring ("declared but never referenced") despite [[maybe_unused]]
      (void) translate_indices;

      item_type items_loc[items_per_thread];
      item_type* items1_shared;
      if constexpr (items_use_bl2sh)
      {
        ::cuda::std::span items1_src{THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(items1_in + keys1_beg),
                                     static_cast<::cuda::std::size_t>(keys1_count_tile)};
        ::cuda::std::span items2_src{THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(items2_in + keys2_beg),
                                     static_cast<::cuda::std::size_t>(keys2_count_tile)};
        ::cuda::std::span items_buffers{storage.items_shared.c_array};
        auto items1_buffer =
          items_buffers.first(block_load_to_shared::template SharedBufferSizeBytes<item_type>(keys1_count_tile));
        auto items2_buffer =
          items_buffers.last(block_load_to_shared::template SharedBufferSizeBytes<item_type>(keys2_count_tile));
        _CCCL_ASSERT(items1_buffer.end() <= items2_buffer.begin(),
                     "Items buffer needs to be appropriately sized (internal)");
        // block_store_keys above uses shared memory, so make sure all threads are done before we write
        __syncthreads();
        items1_shared            = data(load2sh.CopyAsync(items1_buffer, items1_src));
        item_type* items2_shared = data(load2sh.CopyAsync(items2_buffer, items2_src));
        auto token               = load2sh.Commit();
        const int items2_offset  = static_cast<int>(items2_shared - items1_shared);
        translate_indices(items2_offset);
        load2sh.Wait(::cuda::std::move(token));
      }
      else
      {
        {
          auto items1_in_cm = try_make_cache_modified_iterator<Policy::LOAD_MODIFIER>(items1_in);
          auto items2_in_cm = try_make_cache_modified_iterator<Policy::LOAD_MODIFIER>(items2_in);
          merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
            items_loc, items1_in_cm + keys1_beg, items2_in_cm + keys2_beg, keys1_count_tile, keys2_count_tile);
          __syncthreads(); // block_store_keys above uses SMEM, so make sure all threads are done before we write to it
          items1_shared = &storage.items_shared[0];
          if constexpr (keys_use_bl2sh)
          {
            const int items2_offset = keys1_count_tile;
            translate_indices(items2_offset);
          }
          merge_sort::reg_to_shared<threads_per_block>(items1_shared, items_loc);
          __syncthreads();
        }
      }

      // gather items from shared mem
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        items_loc[i] = items1_shared[indices[i]];
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
      consume_tile</* IsFullTile = */ true>(tile_idx, tile_base, items_per_tile);
    }
    else
    {
      consume_tile</* IsFullTile = */ false>(tile_idx, tile_base, items_in_tile);
    }
  }
};
} // namespace detail::merge
CUB_NAMESPACE_END

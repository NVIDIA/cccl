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
namespace detail
{
namespace merge
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
  static constexpr Offset items_per_tile = Policy::ITEMS_PER_TILE;

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

  static constexpr bool keys_use_block_load_to_shared  = use_block_load_to_shared<key_type, KeysIt1, KeysIt2>;
  static constexpr bool items_use_block_load_to_shared = use_block_load_to_shared<item_type, ItemsIt1, ItemsIt2>;
  static constexpr bool need_block_load_to_shared = keys_use_block_load_to_shared || items_use_block_load_to_shared;
  static constexpr int load2sh_minimum_align      = block_load_to_shared::template SharedBufferAlignBytes<char>();

  struct empty_t
  {
    struct TempStorage
    {};
    _CCCL_DEVICE _CCCL_FORCEINLINE empty_t(TempStorage) {}
  };

  using optional_load2sh_t = ::cuda::std::conditional_t<need_block_load_to_shared, block_load_to_shared, empty_t>;

  using keys_load_it1 =
    ::cuda::std::conditional_t<keys_use_block_load_to_shared,
                               KeysIt1,
                               try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, KeysIt1>>;
  using keys_load_it2 =
    ::cuda::std::conditional_t<keys_use_block_load_to_shared,
                               KeysIt2,
                               try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, KeysIt2>>;
  using items_load_it1 =
    ::cuda::std::conditional_t<items_use_block_load_to_shared,
                               ItemsIt1,
                               try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, ItemsIt1>>;
  using items_load_it2 =
    ::cuda::std::conditional_t<items_use_block_load_to_shared,
                               ItemsIt2,
                               try_make_cache_modified_iterator_t<Policy::LOAD_MODIFIER, ItemsIt2>>;

  template <typename ValueT, bool UseBlockLoadToShared>
  struct alignas(UseBlockLoadToShared ? block_load_to_shared::template SharedBufferAlignBytes<ValueT>()
                                      : alignof(ValueT)) buffer_t
  {
    // Need extra bytes of padding for TMA because this static buffer has to hold the two dynamically sized buffers.
    char c_array[UseBlockLoadToShared ? (block_load_to_shared::template SharedBufferSizeBytes<ValueT>(items_per_tile + 1)
                                         + (alignof(ValueT) < load2sh_minimum_align ? 2 * load2sh_minimum_align : 0))
                                      : sizeof(ValueT) * (items_per_tile + 1)];
  };

  struct temp_storages_bl2sh
  {
    union
    {
      typename block_store_keys::TempStorage store_keys;
      typename block_store_items::TempStorage store_items;
      buffer_t<key_type, keys_use_block_load_to_shared> keys_shared;
      buffer_t<item_type, items_use_block_load_to_shared> items_shared;
    };
    typename block_load_to_shared::TempStorage load2sh;
  };

  union temp_storages_fallback
  {
    typename block_store_keys::TempStorage store_keys;
    typename block_store_items::TempStorage store_items;

    buffer_t<key_type, keys_use_block_load_to_shared> keys_shared;
    buffer_t<item_type, items_use_block_load_to_shared> items_shared;

    typename empty_t::TempStorage load2sh;
  };

  using temp_storages =
    ::cuda::std::conditional_t<need_block_load_to_shared, temp_storages_bl2sh, temp_storages_fallback>;

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
  Offset* merge_partitions;

  template <bool IsFullTile>
  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_tile(Offset tile_idx, Offset tile_base, int num_remaining)
  {
    const Offset partition_beg = merge_partitions[tile_idx + 0];
    const Offset partition_end = merge_partitions[tile_idx + 1];

    const Offset diag0 = items_per_tile * tile_idx;
    const Offset diag1 = (::cuda::std::min) (keys1_count + keys2_count, diag0 + items_per_tile);

    // compute bounding box for keys1 & keys2
    const Offset keys1_beg = partition_beg;
    const Offset keys1_end = partition_end;
    const Offset keys2_beg = diag0 - keys1_beg;
    const Offset keys2_end = diag1 - keys1_end;

    // number of keys per tile
    const int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
    const int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

    optional_load2sh_t load2sh{storage.load2sh};

    key_type* keys1_shared;
    key_type* keys2_shared;
    int keys2_offset;
    if constexpr (keys_use_block_load_to_shared)
    {
      ::cuda::std::span keys1_src{THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(keys1_in + keys1_beg),
                                  static_cast<::cuda::std::size_t>(num_keys1)};
      ::cuda::std::span keys2_src{THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(keys2_in + keys2_beg),
                                  static_cast<::cuda::std::size_t>(num_keys2)};
      ::cuda::std::span keys_buffers{storage.keys_shared.c_array};
      auto keys1_buffer = keys_buffers.first(block_load_to_shared::template SharedBufferSizeBytes<key_type>(num_keys1));
      auto keys2_buffer = keys_buffers.last(block_load_to_shared::template SharedBufferSizeBytes<key_type>(num_keys2));
      _CCCL_ASSERT(keys1_buffer.end() <= keys2_buffer.begin(),
                   "Keys buffer needs to be appropriately sized (internal)");
      auto keys1_sh = load2sh.CopyAsync(keys1_buffer, keys1_src);
      auto keys2_sh = load2sh.CopyAsync(keys2_buffer, keys2_src);
      load2sh.Commit();
      keys1_shared = data(keys1_sh);
      keys2_shared = data(keys2_sh);
      // Needed for using keys1_shared as one big buffer including both ranges in SerialMerge
      keys2_offset = static_cast<int>(keys2_shared - keys1_shared);
      load2sh.Wait();
    }
    else
    {
      key_type keys_loc[items_per_thread];
      merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
        keys_loc, keys1_in + keys1_beg, keys2_in + keys2_beg, num_keys1, num_keys2);
      keys1_shared = &::cuda::ptr_rebind<key_type>(storage.keys_shared.c_array)[0];
      // Needed for using keys1_shared as one big buffer including both ranges in SerialMerge
      keys2_offset = num_keys1;
      keys2_shared = keys1_shared + keys2_offset;
      merge_sort::reg_to_shared<threads_per_block>(keys1_shared, keys_loc);
      __syncthreads();
    }

    // use binary search in shared memory to find merge path for each of thread.
    // we can use int type here, because the number of items in shared memory is limited
    const int diag0_loc = (::cuda::std::min) (num_keys1 + num_keys2, static_cast<int>(items_per_thread * threadIdx.x));

    const int keys1_beg_loc = MergePath(keys1_shared, keys2_shared, num_keys1, num_keys2, diag0_loc, compare_op);
    const int keys1_end_loc = num_keys1;
    const int keys2_beg_loc = diag0_loc - keys1_beg_loc;
    const int keys2_end_loc = num_keys2;

    const int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
    const int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

    // perform serial merge
    key_type keys_loc[items_per_thread];
    int indices[items_per_thread];
    cub::SerialMerge(
      keys1_shared,
      keys1_beg_loc,
      keys2_offset + keys2_beg_loc,
      num_keys1_loc,
      num_keys2_loc,
      keys_loc,
      indices,
      compare_op);
    __syncthreads();

    // write keys
    if (IsFullTile)
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

      item_type* items1_shared;
      int items2_offset;
      if constexpr (keys_use_block_load_to_shared)
      {
        ::cuda::std::span items1_src{THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(items1_in + keys1_beg),
                                     static_cast<::cuda::std::size_t>(num_keys1)};
        ::cuda::std::span items2_src{THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(items2_in + keys2_beg),
                                     static_cast<::cuda::std::size_t>(num_keys2)};
        ::cuda::std::span items_buffers{storage.items_shared.c_array};
        auto items1_buffer =
          items_buffers.first(block_load_to_shared::template SharedBufferSizeBytes<item_type>(num_keys1));
        auto items2_buffer =
          items_buffers.last(block_load_to_shared::template SharedBufferSizeBytes<item_type>(num_keys2));
        _CCCL_ASSERT(items1_buffer.end() <= items2_buffer.begin(),
                     "Items buffer needs to be appropriately sized (internal)");
        // block_store_keys above uses shared memory, so make sure all threads are done before we write
        __syncthreads();
        auto items1_sh = load2sh.CopyAsync(items1_buffer, items1_src);
        auto items2_sh = load2sh.CopyAsync(items2_buffer, items2_src);
        load2sh.Commit();
        items1_shared            = data(items1_sh);
        item_type* items2_shared = data(items2_sh);
        items2_offset            = static_cast<int>(items2_shared - items1_shared);
        translate_indices(items2_offset);
        load2sh.Wait();
      }
      else
      {
        item_type items_loc[items_per_thread];
        merge_sort::gmem_to_reg<threads_per_block, IsFullTile>(
          items_loc, items1_in + keys1_beg, items2_in + keys2_beg, num_keys1, num_keys2);
        __syncthreads(); // block_store_keys above uses shared memory, so make sure all threads are done before we write
                         // to it
        items1_shared = &::cuda::ptr_rebind<item_type>(storage.items_shared.c_array)[0];
        items2_offset = num_keys1;
        if constexpr (keys_use_block_load_to_shared)
        {
          translate_indices(items2_offset);
        }
        merge_sort::reg_to_shared<threads_per_block>(items1_shared, items_loc);
        __syncthreads();
      }

      // gather items from shared mem
      item_type items_loc[items_per_thread];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < items_per_thread; ++i)
      {
        items_loc[i] = items1_shared[indices[i]];
      }
      __syncthreads();

      // write from reg to gmem
      if (IsFullTile)
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
    // XXX with 8.5 changing type to Offset (or long long) results in error!
    // TODO(bgruber): is the above still true?
    const int tile_idx     = static_cast<int>(blockIdx.x);
    const Offset tile_base = tile_idx * items_per_tile;
    // TODO(bgruber): random mixing of int and Offset
    const int items_in_tile =
      static_cast<int>((::cuda::std::min) (static_cast<Offset>(items_per_tile), keys1_count + keys2_count - tile_base));
    if (items_in_tile == items_per_tile)
    {
      consume_tile<true>(tile_idx, tile_base, items_per_tile); // full tile
    }
    else
    {
      consume_tile<false>(tile_idx, tile_base, items_in_tile); // partial tile
    }
  }
};
} // namespace merge
} // namespace detail
CUB_NAMESPACE_END

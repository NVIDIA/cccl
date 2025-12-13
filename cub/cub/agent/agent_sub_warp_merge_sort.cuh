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

#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_store.cuh>

#include <nv/target>

CUB_NAMESPACE_BEGIN

template <int BlockThreadsArg,
          int WarpThreadsArg,
          int ItemsPerThreadArg,
          cub::WarpLoadAlgorithm LoadAlgorithmArg   = cub::WARP_LOAD_DIRECT,
          cub::CacheLoadModifier LoadModifierArg    = cub::LOAD_LDG,
          cub::WarpStoreAlgorithm StoreAlgorithmArg = cub::WARP_STORE_DIRECT>
struct AgentSubWarpMergeSortPolicy
{
  static constexpr int BLOCK_THREADS      = BlockThreadsArg;
  static constexpr int WARP_THREADS       = WarpThreadsArg;
  static constexpr int ITEMS_PER_THREAD   = ItemsPerThreadArg;
  static constexpr int ITEMS_PER_TILE     = WARP_THREADS * ITEMS_PER_THREAD;
  static constexpr int SEGMENTS_PER_BLOCK = BLOCK_THREADS / WARP_THREADS;

  static constexpr cub::WarpLoadAlgorithm LOAD_ALGORITHM   = LoadAlgorithmArg;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER    = LoadModifierArg;
  static constexpr cub::WarpStoreAlgorithm STORE_ALGORITHM = StoreAlgorithmArg;
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
namespace detail
{
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  SubWarpMergeSortAgentPolicy,
  (GenericAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (WARP_THREADS, WarpThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (ITEMS_PER_TILE, ItemsPerTile, int),
  (SEGMENTS_PER_BLOCK, SegmentsPerBlock, int),
  (LOAD_ALGORITHM, LoadAlgorithm, cub::WarpLoadAlgorithm),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier),
  (STORE_ALGORITHM, StoreAlgorithm, cub::WarpStoreAlgorithm))
} // namespace detail
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

namespace detail::sub_warp_merge_sort
{
/**
 * @brief AgentSubWarpSort implements a sub-warp merge sort.
 *
 * This agent can work with any power of two number of threads, not exceeding
 * 32. The number of threads is defined in the `PolicyT::WARP_THREADS`. Virtual
 * warp of `PolicyT::WARP_THREADS` will efficiently load data using
 * `PolicyT::LOAD_ALGORITHM`, sort it using `WarpMergeSort`, and store it back
 * using `PolicyT::STORE_ALGORITHM`.
 *
 * @tparam IS_DESCENDING
 *   Whether or not the sorted-order is high-to-low
 *
 * @tparam PolicyT
 *   Chained tuning policy
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <bool IS_DESCENDING, typename PolicyT, typename KeyT, typename ValueT, typename OffsetT>
class AgentSubWarpSort
{
  using traits           = detail::radix::traits_t<KeyT>;
  using bit_ordered_type = typename traits::bit_ordered_type;

  struct BinaryOpT
  {
    template <typename T>
    _CCCL_DEVICE bool operator()(T lhs, T rhs) const noexcept
    {
      if constexpr (IS_DESCENDING)
      {
        return lhs > rhs;
      }
      else
      {
        return lhs < rhs;
      }
      _CCCL_UNREACHABLE();
    }

#if _CCCL_HAS_NVFP16()
    _CCCL_DEVICE bool operator()(__half lhs, __half rhs) const noexcept
    {
      // Need to explicitly cast to float for SM <= 52.
      if constexpr (IS_DESCENDING)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_53, (return __hgt(lhs, rhs);), (return __half2float(lhs) > __half2float(rhs);));
      }
      else
      {
        NV_IF_TARGET(NV_PROVIDES_SM_53, (return __hlt(lhs, rhs);), (return __half2float(lhs) < __half2float(rhs);));
      }
      _CCCL_UNREACHABLE();
    }
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
    _CCCL_DEVICE bool operator()(__nv_bfloat16 lhs, __nv_bfloat16 rhs) const noexcept
    {
      // Need to explicitly cast to float for SM < 80.
      if constexpr (IS_DESCENDING)
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_80, (return __hgt(lhs, rhs);), (return __bfloat162float(lhs) > __bfloat162float(rhs);));
      }
      else
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_80, (return __hlt(lhs, rhs);), (return __bfloat162float(lhs) < __bfloat162float(rhs);));
      }
      _CCCL_UNREACHABLE();
    }
#endif // _CCCL_HAS_NVBF16()
  };

#if _CCCL_HAS_NVFP16()
  _CCCL_DEVICE static bool equal(__half lhs, __half rhs)
  {
    // Need to explicitly cast to float for SM <= 52.
    NV_IF_TARGET(NV_PROVIDES_SM_53, (return __heq(lhs, rhs);), (return __half2float(lhs) == __half2float(rhs);));
  }
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  _CCCL_DEVICE static bool equal(__nv_bfloat16 lhs, __nv_bfloat16 rhs)
  {
    // Need to explicitly cast to float for SM < 80.
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return __heq(lhs, rhs);), (return __bfloat162float(lhs) == __bfloat162float(rhs);));
  }
#endif // _CCCL_HAS_NVBF16()

  template <typename T>
  _CCCL_DEVICE static bool equal(T lhs, T rhs)
  {
    return lhs == rhs;
  }

  _CCCL_DEVICE static bool get_oob_default(::cuda::std::true_type /* is bool */)
  {
    // Traits<KeyT>::MAX_KEY for `bool` is 0xFF which is different from `true` and makes
    // comparison with oob unreliable.
    return !IS_DESCENDING;
  }

  _CCCL_DEVICE static KeyT get_oob_default(::cuda::std::false_type /* is bool */)
  {
    // For FP64 the difference is:
    // Lowest() -> -1.79769e+308 = 00...00b -> TwiddleIn -> -0 = 10...00b
    // LOWEST   -> -nan          = 11...11b -> TwiddleIn ->  0 = 00...00b

    // Segmented sort doesn't support custom types at the moment.
    bit_ordered_type default_key_bits = IS_DESCENDING ? traits::min_raw_binary_key(identity_decomposer_t{})
                                                      : traits::max_raw_binary_key(identity_decomposer_t{});
    return reinterpret_cast<KeyT&>(default_key_bits);
  }

public:
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, cub::NullType>;

  using WarpMergeSortT = WarpMergeSort<KeyT, PolicyT::ITEMS_PER_THREAD, PolicyT::WARP_THREADS, ValueT>;

  using KeysLoadItT  = try_make_cache_modified_iterator_t<PolicyT::LOAD_MODIFIER, const KeyT*>;
  using ItemsLoadItT = try_make_cache_modified_iterator_t<PolicyT::LOAD_MODIFIER, const ValueT*>;

  using WarpLoadKeysT = cub::WarpLoad<KeyT, PolicyT::ITEMS_PER_THREAD, PolicyT::LOAD_ALGORITHM, PolicyT::WARP_THREADS>;
  using WarpLoadItemsT =
    cub::WarpLoad<ValueT, PolicyT::ITEMS_PER_THREAD, PolicyT::LOAD_ALGORITHM, PolicyT::WARP_THREADS>;

  using WarpStoreKeysT =
    cub::WarpStore<KeyT, PolicyT::ITEMS_PER_THREAD, PolicyT::STORE_ALGORITHM, PolicyT::WARP_THREADS>;
  using WarpStoreItemsT =
    cub::WarpStore<ValueT, PolicyT::ITEMS_PER_THREAD, PolicyT::STORE_ALGORITHM, PolicyT::WARP_THREADS>;

  union _TempStorage
  {
    typename WarpLoadKeysT::TempStorage load_keys;
    typename WarpLoadItemsT::TempStorage load_items;
    typename WarpMergeSortT::TempStorage sort;
    typename WarpStoreKeysT::TempStorage store_keys;
    typename WarpStoreItemsT::TempStorage store_items;
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& storage;

  _CCCL_DEVICE _CCCL_FORCEINLINE explicit AgentSubWarpSort(TempStorage& temp_storage)
      : storage(temp_storage.Alias())
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessSegment(
    int segment_size, KeysLoadItT keys_input, KeyT* keys_output, ItemsLoadItT values_input, ValueT* values_output)
  {
    WarpMergeSortT warp_merge_sort(storage.sort);

    if (segment_size < 3)
    {
      ShortCircuit(
        warp_merge_sort.get_linear_tid(),
        segment_size,
        keys_input,
        keys_output,
        values_input,
        values_output,
        BinaryOpT{});
    }
    else
    {
      KeyT keys[PolicyT::ITEMS_PER_THREAD];
      ValueT values[PolicyT::ITEMS_PER_THREAD];

      KeyT oob_default = AgentSubWarpSort::get_oob_default(bool_constant_v<::cuda::std::is_same_v<bool, KeyT>>);

      WarpLoadKeysT(storage.load_keys).Load(keys_input, keys, segment_size, oob_default);
      __syncwarp(warp_merge_sort.get_member_mask());

      if (!KEYS_ONLY)
      {
        WarpLoadItemsT(storage.load_items).Load(values_input, values, segment_size);

        __syncwarp(warp_merge_sort.get_member_mask());
      }

      warp_merge_sort.Sort(keys, values, BinaryOpT{}, segment_size, oob_default);
      __syncwarp(warp_merge_sort.get_member_mask());

      WarpStoreKeysT(storage.store_keys).Store(keys_output, keys, segment_size);

      if (!KEYS_ONLY)
      {
        __syncwarp(warp_merge_sort.get_member_mask());
        WarpStoreItemsT(storage.store_items).Store(values_output, values, segment_size);
      }
    }
  }

private:
  /**
   * This method implements a shortcut for sorting less than three items.
   * Only the first thread of a virtual warp is used for soring.
   */
  template <typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ShortCircuit(
    unsigned int linear_tid,
    OffsetT segment_size,
    KeysLoadItT keys_input,
    KeyT* keys_output,
    ItemsLoadItT values_input,
    ValueT* values_output,
    CompareOpT binary_op)
  {
    if (segment_size == 1)
    {
      if (linear_tid == 0)
      {
        if (keys_input.ptr != keys_output)
        {
          keys_output[0] = keys_input[0];
        }

        if (!KEYS_ONLY)
        {
          if (values_input.ptr != values_output)
          {
            values_output[0] = values_input[0];
          }
        }
      }
    }
    else if (segment_size == 2)
    {
      if (linear_tid == 0)
      {
        KeyT lhs = keys_input[0];
        KeyT rhs = keys_input[1];

        if (equal(lhs, rhs) || binary_op(lhs, rhs))
        {
          keys_output[0] = lhs;
          keys_output[1] = rhs;

          if (!KEYS_ONLY)
          {
            if (values_output != values_input.ptr)
            {
              values_output[0] = values_input[0];
              values_output[1] = values_input[1];
            }
          }
        }
        else
        {
          keys_output[0] = rhs;
          keys_output[1] = lhs;

          if (!KEYS_ONLY)
          {
            // values_output might be an alias for values_input, so
            // we have to use registers here

            const ValueT lhs_val = values_input[0];
            const ValueT rhs_val = values_input[1];

            values_output[0] = rhs_val;
            values_output[1] = lhs_val;
          }
        }
      }
    }
  }
};
} // namespace detail::sub_warp_merge_sort

CUB_NAMESPACE_END

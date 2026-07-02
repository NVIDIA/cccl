// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/concepts>

CUB_NAMESPACE_BEGIN

//! The tuning policy for all algorithms in @ref DeviceMerge.
struct MergePolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  BlockStoreAlgorithm store_algorithm; //!< The @ref BlockStoreAlgorithm used for storing items to global memory
  bool use_bulk_copy_for_keys; //!< Whether to use bulk copy (cp.async.bulk) for loading keys into shared memory
  bool use_bulk_copy_for_values; //!< Whether to use bulk copy (cp.async.bulk) for loading values into shared memory

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const MergePolicy& lhs, const MergePolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier && lhs.store_algorithm == rhs.store_algorithm
        && lhs.use_bulk_copy_for_keys == rhs.use_bulk_copy_for_keys
        && lhs.use_bulk_copy_for_values == rhs.use_bulk_copy_for_values;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const MergePolicy& lhs, const MergePolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const MergePolicy& p)
  {
    return os
        << "MergePolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .load_modifier = " << p.load_modifier
        << ", .store_algorithm = " << p.store_algorithm << ", .use_bulk_copy_for_keys = " << p.use_bulk_copy_for_keys
        << ", .use_bulk_copy_for_values = " << p.use_bulk_copy_for_values << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::merge
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept merge_policy_selector = policy_selector<T, MergePolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;
  int key_align;
  int key_is_trivially_relocatable;
  bool key_iterators_are_contiguous;
  bool key_iterator_value_types_are_the_same;
  int value_size; // if 0, then this is a keys-only policy
  int value_align;
  int value_is_trivially_relocatable;
  bool value_iterators_are_contiguous;
  bool value_iterator_value_types_are_the_same;
  int offset_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> MergePolicy
  {
    const int tune_type_size = key_size + value_size;
    const int ipt_800_plus   = nominal_4B_items_to_items(15, tune_type_size);
    const bool can_bulk_keys = (key_size == key_align) && key_is_trivially_relocatable && key_iterators_are_contiguous
                            && key_iterator_value_types_are_the_same;
    const bool can_bulk_values = (value_size == value_align) && value_is_trivially_relocatable
                              && value_iterators_are_contiguous && value_iterator_value_types_are_the_same;

    if (cc >= ::cuda::compute_capability{10, 0})
    {
      return MergePolicy{512, ipt_800_plus, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, can_bulk_keys, can_bulk_values};
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      const bool should_bl2sh_keys = key_size != 8;
      const bool should_bl2sh_pairs =
        !(key_size == 8 && value_size == 8)
        && (key_size != 16 || (key_size == 16 && value_size == 1 && offset_size == 4)
            || (key_size == 16 && value_size == 8));
      // TODO(bgruber): consider not using a combined should_bl2sh flag. Kept to avoid SASS diffs
      const bool should_bl2sh = value_size == 0 ? should_bl2sh_keys : should_bl2sh_pairs;
      return MergePolicy{
        512,
        ipt_800_plus,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        can_bulk_keys && should_bl2sh,
        can_bulk_values && should_bl2sh};
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      const bool should_bl2sh_keys = key_size < 4;
      const bool should_bl2sh_pairs =
        key_size == 1 || (key_size == 2 && value_size < 4) || (key_size == 4 && value_size == 1);
      // TODO(bgruber): consider not using a combined should_bl2sh flag. Kept to avoid SASS diffs
      const bool should_bl2sh = value_size == 0 ? should_bl2sh_keys : should_bl2sh_pairs;
      return MergePolicy{
        512,
        ipt_800_plus,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        can_bulk_keys && should_bl2sh,
        can_bulk_values && should_bl2sh};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      const int ipt_600 = nominal_4B_items_to_items(15, tune_type_size);
      return MergePolicy{512, ipt_600, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, false, false};
    }

    // default is SM52
    const int ipt_520 = nominal_4B_items_to_items(13, tune_type_size);
    return MergePolicy{512, ipt_520, LOAD_LDG, BLOCK_STORE_WARP_TRANSPOSE, false, false};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(merge_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeysIt1, typename ItemsIt1, typename KeysIt2, typename ItemsIt2, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> MergePolicy
  {
    using key_t  = it_value_t<KeysIt1>;
    using item_t = it_value_t<ItemsIt1>;

    return policy_selector{
      int{sizeof(key_t)},
      int{alignof(key_t)},
      THRUST_NS_QUALIFIER::is_trivially_relocatable_v<key_t>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<KeysIt1> && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<KeysIt2>,
      ::cuda::std::is_same_v<key_t, it_value_t<KeysIt2>>,
      ::cuda::std::is_same_v<item_t, NullType> ? 0 : int{sizeof(item_t)},
      int{alignof(item_t)},
      THRUST_NS_QUALIFIER::is_trivially_relocatable_v<item_t>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<ItemsIt1>
        && THRUST_NS_QUALIFIER::is_contiguous_iterator_v<ItemsIt2>,
      ::cuda::std::is_same_v<item_t, it_value_t<ItemsIt2>>,
      int{sizeof(OffsetT)}}(cc);
  }
};
} // namespace detail::merge

CUB_NAMESPACE_END

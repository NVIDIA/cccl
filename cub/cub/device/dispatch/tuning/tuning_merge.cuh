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

#include <cuda/__device/arch_id.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/concepts>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::merge
{
struct merge_policy
{
  int block_threads;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  bool use_block_load_to_shared;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const merge_policy& lhs, const merge_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier && lhs.store_algorithm == rhs.store_algorithm
        && lhs.use_block_load_to_shared == rhs.use_block_load_to_shared;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const merge_policy& lhs, const merge_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const merge_policy& p)
  {
    return os << "merge_policy { .block_threads = " << p.block_threads << ", .items_per_thread = " << p.items_per_thread
              << ", .load_modifier = " << p.load_modifier << ", .store_algorithm = " << p.store_algorithm
              << ", .use_block_load_to_shared = " << p.use_block_load_to_shared << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept merge_policy_selector = policy_selector<T, merge_policy>;
#endif // _CCCL_HAS_CONCEPTS()

_CCCL_HOST_DEVICE constexpr int nominal_4b_items_to_items(int nominal_4b_items_per_thread, int type_size)
{
  return ::cuda::std::clamp(nominal_4b_items_per_thread * 4 / type_size, 1, nominal_4b_items_per_thread);
}

struct policy_selector
{
  int key_size;
  int value_size; // if 0, then this is a keys-only policy
  int offset_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> merge_policy
  {
    const int tune_type_size = key_size + value_size;
    const int ipt_800_plus   = nominal_4b_items_to_items(15, tune_type_size);

    if (arch >= ::cuda::arch_id::sm_100)
    {
      return merge_policy{512, ipt_800_plus, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, true};
    }

    if (arch >= ::cuda::arch_id::sm_90)
    {
      const bool use_bl2sh_keys = key_size != 8;
      const bool use_bl2sh_pairs =
        !(key_size == 8 && value_size == 8)
        && (key_size != 16 || (key_size == 16 && value_size == 1 && offset_size == 4)
            || (key_size == 16 && value_size == 8));
      return merge_policy{
        512, ipt_800_plus, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, value_size == 0 ? use_bl2sh_keys : use_bl2sh_pairs};
    }

    if (arch >= ::cuda::arch_id::sm_80)
    {
      const bool use_bl2sh_keys = key_size < 4;
      const bool use_bl2sh_pairs =
        key_size == 1 || (key_size == 2 && value_size < 4) || (key_size == 4 && value_size == 1);
      return merge_policy{
        512, ipt_800_plus, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, value_size == 0 ? use_bl2sh_keys : use_bl2sh_pairs};
    }

    if (arch >= ::cuda::arch_id::sm_60)
    {
      const int ipt_600 = nominal_4b_items_to_items(15, tune_type_size);
      return merge_policy{512, ipt_600, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, false};
    }

    // default is SM52
    const int ipt_520 = nominal_4b_items_to_items(13, tune_type_size);
    return merge_policy{512, ipt_520, LOAD_LDG, BLOCK_STORE_WARP_TRANSPOSE, false};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(merge_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> merge_policy
  {
    return policy_selector{
      int{sizeof(KeyT)}, ::cuda::std::is_same_v<ValueT, NullType> ? 0 : int{sizeof(ValueT)}, int{sizeof(OffsetT)}}(
      arch);
  }
};
} // namespace detail::merge

CUB_NAMESPACE_END

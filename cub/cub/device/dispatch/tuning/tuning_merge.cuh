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

#include <cub/agent/agent_merge.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <cuda/__device/arch_id.h>

#if _CCCL_HAS_CONCEPTS()
#  include <cuda/std/concepts>
#endif // _CCCL_HAS_CONCEPTS()

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
  return (::cuda::std::min) (nominal_4b_items_per_thread,
                             (::cuda::std::max) (1, nominal_4b_items_per_thread * 4 / type_size));
}

// TODO(bgruber): remove in CCCL 4.0 when we drop all CUB dispatchers
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub
{
  static constexpr bool has_values = !::cuda::std::is_same_v<ValueT, NullType>;

  using tune_type = char[has_values ? sizeof(KeyT) + sizeof(ValueT) : sizeof(KeyT)];

  struct policy500 : ChainedPolicy<500, policy500, policy500>
  {
    using merge_policy =
      agent_policy_t<256, Nominal4BItemsToItems<tune_type>(11), LOAD_LDG, BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy520 : ChainedPolicy<520, policy520, policy500>
  {
    using merge_policy =
      agent_policy_t<512, Nominal4BItemsToItems<tune_type>(13), LOAD_LDG, BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy600 : ChainedPolicy<600, policy600, policy520>
  {
    using merge_policy =
      agent_policy_t<512, Nominal4BItemsToItems<tune_type>(15), LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE>;
  };

  struct policy800 : ChainedPolicy<800, policy800, policy600>
  {
  private:
    static constexpr bool keys_only      = ::cuda::std::is_same_v<ValueT, NullType>;
    static constexpr bool use_bl2sh_keys = sizeof(KeyT) < 4;
    static constexpr bool use_bl2sh_pairs =
      // enable for Key I8
      sizeof(KeyT) == 1 ||
      // enable for Key I16 when ValueT is I8 or I16
      (sizeof(KeyT) == 2 && sizeof(ValueT) < 4) ||
      // enable for Key I32/F32 when ValueT is I8
      (sizeof(KeyT) == 4 && sizeof(ValueT) == 1);

  public:
    using merge_policy =
      agent_policy_t<512,
                     Nominal4BItemsToItems<tune_type>(15),
                     LOAD_DEFAULT,
                     BLOCK_STORE_WARP_TRANSPOSE,
                     /* UseBlockLoadToShared = */ keys_only ? use_bl2sh_keys : use_bl2sh_pairs>;
  };

  struct policy900 : ChainedPolicy<900, policy900, policy800>
  {
  private:
    static constexpr bool keys_only      = ::cuda::std::is_same_v<ValueT, NullType>;
    static constexpr bool use_bl2sh_keys = sizeof(KeyT) != 8; // disable for Key I64
    static constexpr bool use_bl2sh_pairs =
      // disable for Key I64 Value I64, Key F64 Value I64
      !(sizeof(KeyT) == 8 && sizeof(ValueT) == 8)
      // disable for Key I128
      && (sizeof(KeyT) != 16
          // but re-enable for Key I128 Value I8 Offset I32
          || (sizeof(KeyT) == 16 && sizeof(ValueT) == 1 && sizeof(OffsetT) == 4)
          // but re-enable for Key I128 Value I64
          || (sizeof(KeyT) == 16 && sizeof(ValueT) == 8));

  public:
    using merge_policy =
      agent_policy_t<512,
                     Nominal4BItemsToItems<tune_type>(15),
                     LOAD_DEFAULT,
                     BLOCK_STORE_WARP_TRANSPOSE,
                     /* UseBlockLoadToShared = */ keys_only ? use_bl2sh_keys : use_bl2sh_pairs>;
  };

  struct policy1000 : ChainedPolicy<1000, policy1000, policy900>
  {
    using merge_policy =
      agent_policy_t<512,
                     Nominal4BItemsToItems<tune_type>(15),
                     LOAD_DEFAULT,
                     BLOCK_STORE_WARP_TRANSPOSE,
                     /* UseBlockLoadToShared = */ true>;
  };

  using max_policy = policy1000;
  using MaxPolicy  = policy1000;
};

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

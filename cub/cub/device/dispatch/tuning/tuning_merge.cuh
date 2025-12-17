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
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::merge
{
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
};
} // namespace detail::merge

CUB_NAMESPACE_END

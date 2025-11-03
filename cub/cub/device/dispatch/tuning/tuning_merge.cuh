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
template <typename KeyT, typename ValueT>
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

  using max_policy = policy600;
};
} // namespace detail::merge

CUB_NAMESPACE_END

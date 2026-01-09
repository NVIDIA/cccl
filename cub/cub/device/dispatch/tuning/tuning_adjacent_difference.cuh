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

#include <cub/agent/agent_adjacent_difference.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::adjacent_difference
{
template <typename InputIteratorT, bool MayAlias>
struct policy_hub
{
  using ValueT = it_value_t<InputIteratorT>;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    using AdjacentDifferencePolicy =
      AgentAdjacentDifferencePolicy<128,
                                    Nominal8BItemsToItems<ValueT>(7),
                                    BLOCK_LOAD_WARP_TRANSPOSE,
                                    MayAlias ? LOAD_CA : LOAD_LDG,
                                    BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy500;
};
} // namespace detail::adjacent_difference

CUB_NAMESPACE_END

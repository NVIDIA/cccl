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

#include <cub/agent/agent_for.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{
struct policy_hub_t
{
  struct policy_500_t : ChainedPolicy<500, policy_500_t, policy_500_t>
  {
    using for_policy_t = policy_t<256, 2>;
  };

  using MaxPolicy = policy_500_t;
};
} // namespace detail::for_each

CUB_NAMESPACE_END

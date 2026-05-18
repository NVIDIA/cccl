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

#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{
struct for_policy
{
  int threads_per_block;
  int items_per_thread;

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const for_policy& lhs, const for_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const for_policy& lhs, const for_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const for_policy& policy)
  {
    return os << "for_policy { .threads_per_block = " << policy.threads_per_block
              << ", .items_per_thread = " << policy.items_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept for_policy_selector = policy_selector<T, for_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> for_policy
  {
    return for_policy{256, 2};
  }
};

// TODO(bgruber): remove once we publish the tuning API
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

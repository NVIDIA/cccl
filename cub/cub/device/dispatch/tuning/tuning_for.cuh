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

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{
struct for_policy
{
  int block_threads;
  int items_per_thread;

  _CCCL_API constexpr friend bool operator==(const for_policy& lhs, const for_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread;
  }

  _CCCL_API constexpr friend bool operator!=(const for_policy& lhs, const for_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const for_policy& policy)
  {
    return os << "for_policy { .block_threads = " << policy.block_threads
              << ", .items_per_thread = " << policy.items_per_thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept for_policy_selector = policy_selector<T, for_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> for_policy
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

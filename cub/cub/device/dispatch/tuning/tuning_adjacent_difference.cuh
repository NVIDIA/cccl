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

#include <cuda/__device/arch_id.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::adjacent_difference
{
struct adjacent_difference_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;

  _CCCL_API constexpr friend bool
  operator==(const adjacent_difference_policy& lhs, const adjacent_difference_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm;
  }

  _CCCL_API constexpr friend bool
  operator!=(const adjacent_difference_policy& lhs, const adjacent_difference_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const adjacent_difference_policy& p)
  {
    return os << "adjacent_difference_policy { .block_threads = " << p.block_threads
              << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
              << ", .load_modifier = " << p.load_modifier << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept adjacent_difference_policy_selector = policy_selector<T, adjacent_difference_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int value_type_size;
  bool may_alias;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> adjacent_difference_policy
  {
    return adjacent_difference_policy{
      128,
      nominal_8B_items_to_items(7, value_type_size),
      BLOCK_LOAD_WARP_TRANSPOSE,
      may_alias ? LOAD_CA : LOAD_LDG,
      BLOCK_STORE_WARP_TRANSPOSE};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(adjacent_difference_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename InputIteratorT, bool MayAlias>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> adjacent_difference_policy
  {
    constexpr auto policies = policy_selector{static_cast<int>(sizeof(it_value_t<InputIteratorT>)), MayAlias};
    return policies(arch);
  }
};

// TODO(bgruber): remove in CCCL 4.0 when we drop the adjacent difference dispatchers
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

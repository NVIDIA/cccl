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

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

//! The tuning policy for all algorithms in @ref DeviceAdjacentDifference.
struct AdjacentDifferencePolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  BlockLoadAlgorithm load_algorithm; //!< The @ref BlockLoadAlgorithm used for loading items from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  BlockStoreAlgorithm store_algorithm; //!< The @ref BlockStoreAlgorithm used for storing items to global memory

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const AdjacentDifferencePolicy& lhs, const AdjacentDifferencePolicy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const AdjacentDifferencePolicy& lhs, const AdjacentDifferencePolicy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const AdjacentDifferencePolicy& p)
  {
    return os << "AdjacentDifferencePolicy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
              << ", .load_modifier = " << p.load_modifier << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::adjacent_difference
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept adjacent_difference_policy_selector = policy_selector<T, AdjacentDifferencePolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int value_type_size;
  bool may_alias;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> AdjacentDifferencePolicy
  {
    return AdjacentDifferencePolicy{
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
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> AdjacentDifferencePolicy
  {
    constexpr auto policies = policy_selector{static_cast<int>(sizeof(it_value_t<InputIteratorT>)), MayAlias};
    return policies(cc);
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
      agent_adjacent_difference_policy<128,
                                       Nominal8BItemsToItems<ValueT>(7),
                                       BLOCK_LOAD_WARP_TRANSPOSE,
                                       MayAlias ? LOAD_CA : LOAD_LDG,
                                       BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy500;
};
} // namespace detail::adjacent_difference

CUB_NAMESPACE_END

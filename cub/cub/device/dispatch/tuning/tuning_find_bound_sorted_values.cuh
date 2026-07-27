// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/concepts>

#if _CCCL_HOSTED()
#  include <ostream>
#endif // _CCCL_HOSTED()

CUB_NAMESPACE_BEGIN

//! The tuning policy for the LowerBoundSortedValues and UpperBoundSortedValues algorithms in @ref DeviceFind.
struct FindBoundSortedValuesPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const FindBoundSortedValuesPolicy& lhs, const FindBoundSortedValuesPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const FindBoundSortedValuesPolicy& lhs, const FindBoundSortedValuesPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const FindBoundSortedValuesPolicy& p)
  {
    return os << "FindBoundSortedValuesPolicy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::find_bound_sorted_values
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept find_bound_sorted_values_policy_selector = policy_selector<T, FindBoundSortedValuesPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int range_type_size;
  int values_type_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> FindBoundSortedValuesPolicy
  {
    const int combined_size = range_type_size + values_type_size;
    const int ipt           = cub::detail::nominal_4B_items_to_items(15, combined_size);

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      return FindBoundSortedValuesPolicy{512, ipt, LOAD_DEFAULT};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      return FindBoundSortedValuesPolicy{256, ipt, LOAD_DEFAULT};
    }

    // default
    return FindBoundSortedValuesPolicy{256, ipt, LOAD_LDG};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(find_bound_sorted_values_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename RangeT, typename ValuesT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> FindBoundSortedValuesPolicy
  {
    return policy_selector{int{sizeof(RangeT)}, int{sizeof(ValuesT)}}(cc);
  }
};
} // namespace detail::find_bound_sorted_values

CUB_NAMESPACE_END

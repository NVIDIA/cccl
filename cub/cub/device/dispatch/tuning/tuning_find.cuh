// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/concepts>

CUB_NAMESPACE_BEGIN

//! The tuning policy for the @c FindIf algorithms in @ref DeviceFind.
struct FindIfPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  int vec_size; //!< Vectorization size for loading items
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const FindIfPolicy& lhs, const FindIfPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vec_size == rhs.vec_size && lhs.load_modifier == rhs.load_modifier;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const FindIfPolicy& lhs, const FindIfPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const FindIfPolicy& p)
  {
    return os
        << "FindIfPolicy { .threads_per_block = " << p.threads_per_block << ", .items_per_thread = "
        << p.items_per_thread << ", .vec_size = " << p.vec_size << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::find
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept find_policy_selector = policy_selector<T, FindIfPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int input_type_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> FindIfPolicy
  {
    // FindIfPolicy (GTX670: 154.0 @ 48M 4B items) - single policy for all ccs
    const auto scaled = scale_mem_bound(128, 16, input_type_size);
    return FindIfPolicy{scaled.threads_per_block, scaled.items_per_thread, 4, LOAD_LDG};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(find_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename InputType>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> FindIfPolicy
  {
    return policy_selector{int{sizeof(InputType)}}(cc);
  }
};
} // namespace detail::find

CUB_NAMESPACE_END

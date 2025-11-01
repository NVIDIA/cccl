// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>

THRUST_NAMESPACE_BEGIN

// Forward declarations
namespace system::detail::sequential
{
struct tag;
template <class>
struct execution_policy;
} // namespace system::detail::sequential

namespace detail
{

// Helper to determine if NVTX should be enabled for a given policy
// NVTX is DISABLED only for thrust::seq and any policy derived from sequential::execution_policy
// ENABLED for all other policies (CUDA, OMP, TBB, etc.)
template <typename DerivedPolicy>
inline constexpr bool should_enable_nvtx_for_policy()
{
  using Policy = typename ::cuda::std::remove_cv<typename ::cuda::std::remove_reference<DerivedPolicy>::type>::type;

  // Check if Policy is derived from sequential::execution_policy
  // This catches thrust::seq, cpp::tag, and any other sequential-based policy
  return !::cuda::std::is_base_of<thrust::system::detail::sequential::execution_policy<Policy>, Policy>::value
      && !::cuda::std::is_same<Policy, thrust::system::detail::sequential::tag>::value;
}

} // namespace detail

THRUST_NAMESPACE_END

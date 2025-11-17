// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_base_of.h>

THRUST_NAMESPACE_BEGIN

// Forward declarations
namespace system::detail::sequential
{
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
  using Policy = ::cuda::std::decay_t<DerivedPolicy>;
  // This catches thrust::seq, cpp::tag, and any other sequential-based policy
  return !::cuda::std::is_base_of_v<thrust::system::detail::sequential::execution_policy<Policy>, Policy>;
}
} // namespace detail

THRUST_NAMESPACE_END

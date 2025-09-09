// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
struct tag;

template <typename>
struct execution_policy;

template <>
struct execution_policy<tag> : thrust::execution_policy<tag>
{};

struct tag : execution_policy<tag>
{};

// allow conversion to tag when it is not a successor
template <typename Derived>
struct execution_policy : thrust::execution_policy<Derived>
{
  // allow conversion to tag
  _CCCL_HOST_DEVICE operator tag() const
  {
    return {};
  }
};

// TODO(bgruber): do we need this global variable? We already have thrust::seq
_CCCL_GLOBAL_CONSTANT tag seq;

} // namespace system::detail::sequential
THRUST_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

struct any_system_tag : execution_policy<any_system_tag>
{
  // allow any_system_tag to convert to any type at all
  // XXX make this safer using enable_if<is_tag<T>> upon c++11
  template <typename T>
  _CCCL_HOST_DEVICE operator T() const
  {
    return T();
  }
};

THRUST_NAMESPACE_END

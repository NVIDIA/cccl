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

THRUST_NAMESPACE_BEGIN
namespace detail
{
// a type which may be assigned any other type
struct any_assign
{
  any_assign() = default;

  template <typename T>
  _CCCL_HOST_DEVICE any_assign(T)
  {}

  template <typename T>
  _CCCL_HOST_DEVICE any_assign& operator=(T)
  {
    return *this;
  }
};
} // namespace detail
THRUST_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * (C) Copyright John Maddock 2000.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/preprocessor.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename, bool x>
struct depend_on_instantiation
{
  static constexpr bool value = x;
};

//! Deprecated [Since 3.0]
#define THRUST_STATIC_ASSERT(B) static_assert(B)
//! Deprecated [Since 3.0]
#define THRUST_STATIC_ASSERT_MSG(B, msg) static_assert(B, msg)
} // namespace detail

THRUST_NAMESPACE_END

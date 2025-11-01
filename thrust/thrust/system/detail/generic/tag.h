// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file generic/tag.h
 *  \brief Implementation of the generic backend's tag.
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

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

// tag exists only to make the generic entry points the least priority match
// during ADL. tag should not be derived from and is constructible from anything
struct tag
{
  template <typename T>
  _CCCL_HOST_DEVICE inline tag(const T&)
  {}
};

} // namespace system::detail::generic
THRUST_NAMESPACE_END

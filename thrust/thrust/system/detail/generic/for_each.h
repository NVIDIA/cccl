// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file for_each.h
 *  \brief Generic implementation of for_each & for_each_n.
 *         It is an error to call these functions; they have no implementation.
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
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/static_assert.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

template <typename DerivedPolicy, typename InputIterator, typename UnaryFunction>
_CCCL_HOST_DEVICE InputIterator
for_each(thrust::execution_policy<DerivedPolicy>&, InputIterator first, InputIterator, UnaryFunction)
{
  static_assert(thrust::detail::depend_on_instantiation<InputIterator, false>::value, "unimplemented for this system");
  return first;
} // end for_each()

template <typename DerivedPolicy, typename InputIterator, typename Size, typename UnaryFunction>
_CCCL_HOST_DEVICE InputIterator
for_each_n(thrust::execution_policy<DerivedPolicy>&, InputIterator first, Size, UnaryFunction)
{
  static_assert(thrust::detail::depend_on_instantiation<InputIterator, false>::value, "unimplemented for this system");
  return first;
} // end for_each_n()

} // namespace system::detail::generic
THRUST_NAMESPACE_END

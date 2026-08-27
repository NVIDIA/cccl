// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file for_each.h
 *  \brief Sequential implementations of for_each functions.
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
#include <thrust/detail/function.h>
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/__algorithm/for_each.h>
#include <cuda/std/__algorithm/for_each_n.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename UnaryFunction>
_CCCL_HOST_DEVICE InputIterator
for_each(sequential::execution_policy<DerivedPolicy>&, InputIterator first, InputIterator last, UnaryFunction f)
{
  ::cuda::std::for_each(first, last, thrust::detail::wrapped_function<UnaryFunction>{f});
  return last;
} // end for_each()

template <typename DerivedPolicy, typename InputIterator, typename Size, typename UnaryFunction>
_CCCL_HOST_DEVICE InputIterator
for_each_n(sequential::execution_policy<DerivedPolicy>&, InputIterator first, Size n, UnaryFunction f)
{
  return ::cuda::std::for_each_n(first, n, thrust::detail::wrapped_function<UnaryFunction>{f});
} // end for_each_n()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

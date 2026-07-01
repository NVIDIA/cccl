// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file find.h
 *  \brief Sequential implementation of find_if.
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

#include <cuda/std/__algorithm/find_if.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE InputIterator
find_if(execution_policy<DerivedPolicy>&, InputIterator first, InputIterator last, Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};
  return ::cuda::std::find_if(first, last, wrapped_pred);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file find.h
 *  \brief OpenMP implementation of find_if.
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

#include <thrust/system/detail/generic/find.h>
#include <thrust/system/omp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename DerivedPolicy, typename InputIterator, typename Predicate>
InputIterator find_if(execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  // omp prefers generic::find_if to cpp::find_if
  return thrust::system::detail::generic::find_if(exec, first, last, pred);
}
} // end namespace system::omp::detail
THRUST_NAMESPACE_END

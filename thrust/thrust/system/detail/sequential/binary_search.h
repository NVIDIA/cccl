// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file binary_search.h
 *  \brief Sequential implementation of binary search algorithms.
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

#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ForwardIterator lower_bound(
  sequential::execution_policy<DerivedPolicy>&,
  ForwardIterator first,
  ForwardIterator last,
  const T& val,
  StrictWeakOrdering comp)
{
  return ::cuda::std::lower_bound(first, last, val, thrust::detail::wrapped_function<StrictWeakOrdering>{comp});
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ForwardIterator upper_bound(
  sequential::execution_policy<DerivedPolicy>&,
  ForwardIterator first,
  ForwardIterator last,
  const T& val,
  StrictWeakOrdering comp)
{
  return ::cuda::std::upper_bound(first, last, val, thrust::detail::wrapped_function<StrictWeakOrdering>{comp});
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE bool binary_search(
  sequential::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const T& val,
  StrictWeakOrdering comp)
{
  ForwardIterator iter = sequential::lower_bound(exec, first, last, val, comp);

  thrust::detail::wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  return iter != last && !wrapped_comp(val, *iter);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

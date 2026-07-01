// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file set_operations.h
 *  \brief Sequential implementation of set operation functions.
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

#include <cuda/std/__algorithm/set_difference.h>
#include <cuda/std/__algorithm/set_intersection.h>
#include <cuda/std/__algorithm/set_symmetric_difference.h>
#include <cuda/std/__algorithm/set_union.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  // wrap comp
  const thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};
  return ::cuda::std::set_difference(first1, last1, first2, last2, result, wrapped_comp);
} // end set_difference()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  // wrap comp
  const thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};
  return ::cuda::std::set_intersection(first1, last1, first2, last2, result, wrapped_comp);
} // end set_intersection()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  // wrap comp
  const thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};
  return ::cuda::std::set_symmetric_difference(first1, last1, first2, last2, result, wrapped_comp);
} // end set_symmetric_difference()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator set_union(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  // wrap comp
  const thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};
  return ::cuda::std::set_union(first1, last1, first2, last2, result, wrapped_comp);
} // end set_union()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

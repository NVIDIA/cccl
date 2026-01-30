// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file remove.h
 *  \brief Sequential implementations of remove functions.
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

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator
remove_if(sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};

  // advance iterators until wrapped_pred(*first) is true or we reach the end of input
  while (first != last && !wrapped_pred(*first))
  {
    ++first;
  }

  if (first == last)
  {
    return first;
  }

  // result always trails first
  ForwardIterator result = first;

  ++first;

  while (first != last)
  {
    if (!wrapped_pred(*first))
    {
      *result = *first;
      ++result;
    }
    ++first;
  }

  return result;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator remove_if(
  sequential::execution_policy<DerivedPolicy>&,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};

  // advance iterators until wrapped_pred(*stencil) is true or we reach the end of input
  while (first != last && !wrapped_pred(*stencil))
  {
    ++first;
    ++stencil;
  }

  if (first == last)
  {
    return first;
  }

  // result always trails first
  ForwardIterator result = first;

  ++first;
  ++stencil;

  while (first != last)
  {
    if (!wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    }
    ++first;
    ++stencil;
  }

  return result;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
_CCCL_HOST_DEVICE OutputIterator remove_copy_if(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};

  while (first != last)
  {
    if (!wrapped_pred(*first))
    {
      *result = *first;
      ++result;
    }

    ++first;
  }

  return result;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
_CCCL_HOST_DEVICE OutputIterator remove_copy_if(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred)
{
  // wrap pred
  thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};

  while (first != last)
  {
    if (!wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    }

    ++first;
    ++stencil;
  }

  return result;
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

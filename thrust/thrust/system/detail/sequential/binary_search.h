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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>

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
  // wrap comp
  thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};

  using difference_type = thrust::detail::it_difference_t<ForwardIterator>;

  difference_type len = ::cuda::std::distance(first, last);

  while (len > 0)
  {
    difference_type half   = len >> 1;
    ForwardIterator middle = first;

    ::cuda::std::advance(middle, half);

    if (wrapped_comp(*middle, val))
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
    else
    {
      len = half;
    }
  }

  return first;
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
  // wrap comp
  thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};

  using difference_type = thrust::detail::it_difference_t<ForwardIterator>;

  difference_type len = ::cuda::std::distance(first, last);

  while (len > 0)
  {
    difference_type half   = len >> 1;
    ForwardIterator middle = first;

    ::cuda::std::advance(middle, half);

    if (wrapped_comp(val, *middle))
    {
      len = half;
    }
    else
    {
      first = middle;
      ++first;
      len = len - half - 1;
    }
  }

  return first;
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

  // wrap comp
  thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};

  return iter != last && !wrapped_comp(val, *iter);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

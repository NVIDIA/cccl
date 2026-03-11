// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file extrema.h
 *  \brief Sequential implementations of extrema functions.
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

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator min_element(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // wrap comp
  thrust::detail::wrapped_function<BinaryPredicate, bool> wrapped_comp{comp};

  ForwardIterator imin = first;

  for (; first != last; ++first)
  {
    if (wrapped_comp(*first, *imin))
    {
      imin = first;
    }
  }

  return imin;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator max_element(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // wrap comp
  thrust::detail::wrapped_function<BinaryPredicate, bool> wrapped_comp{comp};

  ForwardIterator imax = first;

  for (; first != last; ++first)
  {
    if (wrapped_comp(*imax, *first))
    {
      imax = first;
    }
  }

  return imax;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // wrap comp
  thrust::detail::wrapped_function<BinaryPredicate, bool> wrapped_comp{comp};

  ForwardIterator imin = first;
  ForwardIterator imax = first;

  for (; first != last; ++first)
  {
    if (wrapped_comp(*first, *imin))
    {
      imin = first;
    }

    if (wrapped_comp(*imax, *first))
    {
      imax = first;
    }
  }

  return ::cuda::std::make_pair(imin, imax);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

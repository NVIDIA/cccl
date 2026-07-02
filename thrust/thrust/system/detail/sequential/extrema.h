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

#include <cuda/std/__algorithm/max_element.h>
#include <cuda/std/__algorithm/min_element.h>
#include <cuda/std/__algorithm/minmax_element.h>
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
  const thrust::detail::wrapped_function<BinaryPredicate, bool> wrapped_comp{comp};
  return ::cuda::std::min_element(first, last, wrapped_comp);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator max_element(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // wrap comp
  const thrust::detail::wrapped_function<BinaryPredicate, bool> wrapped_comp{comp};
  return ::cuda::std::max_element(first, last, wrapped_comp);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // wrap comp
  const thrust::detail::wrapped_function<BinaryPredicate, bool> wrapped_comp{comp};
  return ::cuda::std::minmax_element(first, last, wrapped_comp);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

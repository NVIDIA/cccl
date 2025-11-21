// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/system/detail/generic/extrema.h>
#include <thrust/system/omp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator
max_element(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // omp prefers generic::max_element to cpp::max_element
  return thrust::system::detail::generic::max_element(exec, first, last, comp);
} // end max_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator
min_element(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // omp prefers generic::min_element to cpp::min_element
  return thrust::system::detail::generic::min_element(exec, first, last, comp);
} // end min_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
::cuda::std::pair<ForwardIterator, ForwardIterator>
minmax_element(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  // omp prefers generic::minmax_element to cpp::minmax_element
  return thrust::system::detail::generic::minmax_element(exec, first, last, comp);
} // end minmax_element()
} // end namespace system::omp::detail
THRUST_NAMESPACE_END

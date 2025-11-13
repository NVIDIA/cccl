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
#include <thrust/system/detail/generic/unique.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator
unique(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
{
  // tbb prefers generic::unique to cpp::unique
  return thrust::system::detail::generic::unique(exec, first, last, binary_pred);
} // end unique()

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred)
{
  // tbb prefers generic::unique_copy to cpp::unique_copy
  return thrust::system::detail::generic::unique_copy(exec, first, last, output, binary_pred);
} // end unique_copy()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
thrust::detail::it_difference_t<ForwardIterator> unique_count(
  execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
{
  // tbb prefers generic::unique_count to cpp::unique_count
  return thrust::system::detail::generic::unique_count(exec, first, last, binary_pred);
} // end unique_count()
} // end namespace system::tbb::detail
THRUST_NAMESPACE_END

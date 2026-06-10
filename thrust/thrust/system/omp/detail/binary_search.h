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

#include <thrust/system/detail/generic/binary_search.h>
#include <thrust/system/omp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(
  execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  // omp prefers generic::lower_bound to cpp::lower_bound
  return thrust::system::detail::generic::lower_bound(exec, begin, end, value, comp);
}

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering, typename Backend>
ForwardIterator upper_bound(
  execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  // omp prefers generic::upper_bound to cpp::upper_bound
  return thrust::system::detail::generic::upper_bound(exec, begin, end, value, comp);
}

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(execution_policy<DerivedPolicy>& exec,
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value,
                   StrictWeakOrdering comp)
{
  // omp prefers generic::binary_search to cpp::binary_search
  return thrust::system::detail::generic::binary_search(exec, begin, end, value, comp);
}
} // end namespace system::omp::detail
THRUST_NAMESPACE_END

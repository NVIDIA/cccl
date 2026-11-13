// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file binary_search.h
 *  \brief Generic implementations of binary search functions.
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
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE_API ForwardIterator
lower_bound(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value);

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API ForwardIterator lower_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE_API ForwardIterator
upper_bound(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value);

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API ForwardIterator upper_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE_API bool binary_search(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value);

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API bool binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE_API OutputIterator lower_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output);

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API OutputIterator lower_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE_API OutputIterator upper_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output);

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API OutputIterator upper_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE_API OutputIterator binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output);

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API OutputIterator binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
_CCCL_HOST_DEVICE_API ::cuda::std::pair<ForwardIterator, ForwardIterator> equal_range(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const LessThanComparable& value);

template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE_API ::cuda::std::pair<ForwardIterator, ForwardIterator> equal_range(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const LessThanComparable& value,
  StrictWeakOrdering comp);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/binary_search.inl>

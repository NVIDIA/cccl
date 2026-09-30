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
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
sort(thrust::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last);

template <typename DerivedPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void
sort(thrust::execution_policy<DerivedPolicy>& exec,
     RandomAccessIterator first,
     RandomAccessIterator last,
     StrictWeakOrdering comp);

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
_CCCL_HOST_DEVICE void sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first);

template <typename DerivedPolicy,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
stable_sort(thrust::execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last);

// XXX it is an error to call this function; it has no implementation
template <typename DerivedPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void stable_sort(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator first,
  RandomAccessIterator last,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename RandomAccessIterator1, typename RandomAccessIterator2>
_CCCL_HOST_DEVICE void stable_sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first);

// XXX it is an error to call this function; it has no implementation
template <typename DerivedPolicy,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void stable_sort_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 keys_first,
  RandomAccessIterator1 keys_last,
  RandomAccessIterator2 values_first,
  StrictWeakOrdering comp);

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE bool
is_sorted(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last);

template <typename DerivedPolicy, typename ForwardIterator, typename Compare>
_CCCL_HOST_DEVICE bool
is_sorted(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Compare comp);

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator
is_sorted_until(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last);

template <typename DerivedPolicy, typename ForwardIterator, typename Compare>
_CCCL_HOST_DEVICE ForwardIterator is_sorted_until(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Compare comp);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/sort.inl>

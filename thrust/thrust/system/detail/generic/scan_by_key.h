// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file scan_by_key.h
 *  \brief Generic implementations of key-value scans.
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
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op);

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result);

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/scan_by_key.inl>

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
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator
unique(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last);

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator unique(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator unique_copy(
  thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator output);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator unique_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<ForwardIterator>
unique_count(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last);

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<ForwardIterator> unique_count(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/unique.inl>

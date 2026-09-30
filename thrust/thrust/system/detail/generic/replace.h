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
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
_CCCL_HOST_DEVICE OutputIterator replace_copy_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred,
  const T& new_value);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate,
          typename T>
_CCCL_HOST_DEVICE OutputIterator replace_copy_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred,
  const T& new_value);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator replace_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  const T& old_value,
  const T& new_value);

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate, typename T>
_CCCL_HOST_DEVICE void replace_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  Predicate pred,
  const T& new_value);

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
_CCCL_HOST_DEVICE void replace_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred,
  const T& new_value);

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void
replace(thrust::execution_policy<DerivedPolicy>& exec,
        ForwardIterator first,
        ForwardIterator last,
        const T& old_value,
        const T& new_value);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/replace.inl>

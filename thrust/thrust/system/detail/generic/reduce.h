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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename InputIterator>
_CCCL_HOST_DEVICE thrust::detail::it_value_t<InputIterator>
reduce(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last);

template <typename DerivedPolicy, typename InputIterator, typename T>
_CCCL_HOST_DEVICE T
reduce(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, T init);

template <typename DerivedPolicy, typename InputIterator, typename T, typename BinaryFunction>
_CCCL_HOST_DEVICE T reduce(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  T init,
  BinaryFunction binary_op);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE void reduce_into(
  thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator output);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE void reduce_into(
  thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator output, T init);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
_CCCL_HOST_DEVICE void reduce_into(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  T init,
  BinaryFunction binary_op);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/reduce.inl>

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
template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform_inclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  BinaryFunction binary_op);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator transform_inclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  BinaryFunction binary_op);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator transform_exclusive_scan(
  thrust::execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/transform_scan.inl>

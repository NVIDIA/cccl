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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/system/detail/generic/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cuda/std/functional>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy,
          typename InputIterator,
          typename UnaryFunction,
          typename OutputType,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputType transform_reduce(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  UnaryFunction unary_op,
  OutputType init,
  BinaryFunction binary_op)
{
  thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_first(first, unary_op);
  thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_last(last, unary_op);

  return thrust::reduce(exec, xfrm_first, xfrm_last, init, binary_op);
} // end transform_reduce()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename T,
          typename BinaryOp1,
          typename BinaryOp2>
_CCCL_HOST_DEVICE T transform_reduce(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  T init,
  BinaryOp1 reduce,
  BinaryOp2 transform)
{
  // Create a zip iterator to iterate over both input ranges simultaneously
  const auto first = thrust::make_zip_iterator(first1, first2);
  const auto last  = thrust::make_zip_iterator(last1, first2); // only first iterator matters

  // Use the unary transform_reduce with the zipped iterators and a zip_function
  return thrust::transform_reduce(exec, first, last, thrust::make_zip_function(transform), init, reduce);
} // end transform_reduce()
} // namespace system::detail::generic
THRUST_NAMESPACE_END

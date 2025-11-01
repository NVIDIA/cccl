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
#include <thrust/reduce.h>
#include <thrust/system/detail/generic/tag.h>

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

} // namespace system::detail::generic
THRUST_NAMESPACE_END

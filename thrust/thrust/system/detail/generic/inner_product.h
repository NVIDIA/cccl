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
#include <thrust/detail/internal_functional.h>
#include <thrust/functional.h>
#include <thrust/system/detail/generic/tag.h>
#include <thrust/transform_reduce.h>
#include <thrust/zip_function.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType>
_CCCL_HOST_DEVICE OutputType inner_product(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputType init)
{
  ::cuda::std::plus<OutputType> binary_op1;
  ::cuda::std::multiplies<OutputType> binary_op2;
  return thrust::inner_product(exec, first1, last1, first2, init, binary_op1, binary_op2);
} // end inner_product()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputType,
          typename BinaryFunction1,
          typename BinaryFunction2>
_CCCL_HOST_DEVICE OutputType inner_product(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputType init,
  BinaryFunction1 binary_op1,
  BinaryFunction2 binary_op2)
{
  const auto first = thrust::make_zip_iterator(first1, first2);
  const auto last  = thrust::make_zip_iterator(last1, first2); // only first iterator matters
  return thrust::transform_reduce(exec, first, last, thrust::make_zip_function(binary_op2), init, binary_op1);
} // end inner_product()

} // namespace system::detail::generic
THRUST_NAMESPACE_END

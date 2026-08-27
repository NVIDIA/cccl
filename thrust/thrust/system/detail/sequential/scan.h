// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file scan.h
 *  \brief Sequential implementations of scan functions.
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
#include <thrust/detail/function.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__numeric/inclusive_scan.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  // Use the input iterator's value type per https://wg21.link/P0571
  using ValueType = thrust::detail::it_value_t<InputIterator>;

  return ::cuda::std::inclusive_scan(
    first, last, result, thrust::detail::wrapped_function<BinaryFunction, ValueType>{binary_op});
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  using ValueType = InitialValueType;

  return ::cuda::std::inclusive_scan(
    first, last, result, thrust::detail::wrapped_function<BinaryFunction, ValueType>{binary_op}, init);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  using ValueType = InitialValueType;

  return ::cuda::std::exclusive_scan(
    first, last, result, init, thrust::detail::wrapped_function<BinaryFunction, ValueType>{binary_op});
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

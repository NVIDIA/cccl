// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file adjacent_difference.h
 *  \brief Sequential implementation of adjacent_difference.
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

#include <cuda/std/__numeric/adjacent_difference.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator adjacent_difference(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  using InputType = thrust::detail::it_value_t<InputIterator>;
  // wrap binary_op to handle proxy references
  thrust::detail::wrapped_function<BinaryFunction, InputType> wrapped_op{binary_op};
  return ::cuda::std::adjacent_difference(first, last, result, wrapped_op);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

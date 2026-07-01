// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file reduce.h
 *  \brief Sequential implementation of reduce algorithm.
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
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/__numeric/accumulate.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputType, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputType reduce(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator begin,
  InputIterator end,
  OutputType init,
  BinaryFunction binary_op)
{
  // wrap binary_op
  const thrust::detail::wrapped_function<BinaryFunction, OutputType> wrapped_binary_op{binary_op};
  return ::cuda::std::accumulate(begin, end, init, wrapped_binary_op);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END

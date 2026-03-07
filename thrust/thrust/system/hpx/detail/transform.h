// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file transform.h
 *  \brief HPX implementation of transform.
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
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <hpx/parallel/algorithms/transform.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction op)
{
  // wrap op
  hpx_wrapped_function<UnaryFunction> wrapped_op{op};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::transform(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::transform(first, last, result, wrapped_op);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction>
OutputIterator transform(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op)
{
  // wrap op
  hpx_wrapped_function<BinaryFunction> wrapped_op{op};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator1, ::hpx::forward_traversal_tag>
                && ::hpx::traits::belongs_to_iterator_traversal_v<InputIterator2, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::transform(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::transform(first1, last1, first2, result, wrapped_op);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END

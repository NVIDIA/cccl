// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file merge.h
 *   \brief HPX implementation of adjacent_difference.
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

#include <thrust/system/detail/generic/adjacent_difference.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__numeric/adjacent_difference.h>

#include <hpx/parallel/algorithms/adjacent_difference.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result)
{
  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>
                && ::hpx::traits::belongs_to_iterator_traversal_v<OutputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::adjacent_difference(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result));
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::cuda::std::adjacent_difference(first, last, result);
  }
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  auto wrapped_op = hpx_wrapped_function<BinaryFunction>{binary_op};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>
                && ::hpx::traits::belongs_to_iterator_traversal_v<OutputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::adjacent_difference(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::cuda::std::adjacent_difference(first, last, result, wrapped_op);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END

// this system inherits adjacent_difference
#include <thrust/system/cpp/detail/adjacent_difference.h>

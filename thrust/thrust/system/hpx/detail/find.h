// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file find.h
 *  \brief HPX implementation of find, find_if, and find_if_not.
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

#include <hpx/parallel/algorithms/find.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename InputIterator, typename T>
InputIterator
find(execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, const T& value)
{
  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::find(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      value);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::find(first, last, value);
  }
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
InputIterator
find_if(execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, Predicate pred)
{
  // wrap
  hpx_wrapped_function<Predicate> wrapped_pred(pred);

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::find_if(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::find_if(first, last, wrapped_pred);
  }
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
InputIterator find_if_not(
  execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, Predicate pred)
{
  // wrap
  hpx_wrapped_function<Predicate> wrapped_pred(pred);

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::find_if_not(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::find_if_not(first, last, wrapped_pred);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END

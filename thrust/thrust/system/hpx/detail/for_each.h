// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file for_each.h
 *  \brief HPX implementation of for_each/for_each_n.
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

#include <hpx/parallel/algorithms/for_each.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename InputIterator, typename UnaryFunction>
InputIterator for_each(
  execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, UnaryFunction f)
{
  // wrap f
  hpx_wrapped_function<UnaryFunction> wrapped_f{f};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    (void) ::hpx::for_each(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_f);
  }
  else
  {
    (void) ::hpx::for_each(first, last, wrapped_f);
  }

  return last;
}

template <typename DerivedPolicy, typename InputIterator, typename Size, typename UnaryFunction>
InputIterator
for_each_n(execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, Size n, UnaryFunction f)
{
  // wrap f
  hpx_wrapped_function<UnaryFunction> wrapped_f{f};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::for_each_n(
      hpx::detail::to_hpx_execution_policy(exec), ::thrust::try_unwrap_contiguous_iterator(first), n, wrapped_f);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::for_each_n(first, n, wrapped_f);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END

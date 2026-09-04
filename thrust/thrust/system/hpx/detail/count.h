// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file count.h
 *  \brief HPX implementation of count/count_if.
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

#include <hpx/parallel/algorithms/count.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename InputIterator, typename EqualityComparable>
typename thrust::iterator_traits<InputIterator>::difference_type
count(execution_policy<DerivedPolicy>& exec [[maybe_unused]],
      InputIterator first,
      InputIterator last,
      const EqualityComparable& value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    return ::hpx::count(hpx::detail::to_hpx_execution_policy(exec),
                        ::thrust::try_unwrap_contiguous_iterator(first),
                        ::thrust::try_unwrap_contiguous_iterator(last),
                        value);
  }
  else
  {
    return ::hpx::count(first, last, value);
  }
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
typename thrust::iterator_traits<InputIterator>::difference_type count_if(
  execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, Predicate pred)
{
  // wrap pred
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    return ::hpx::count_if(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
  }
  else
  {
    return ::hpx::count_if(first, last, wrapped_pred);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END

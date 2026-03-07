// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file fill.h
 *  \brief HPX implementation of fill/fill_n.
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

#include <hpx/parallel/algorithms/fill.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename ForwardIterator, typename T>
void fill(
  execution_policy<DerivedPolicy>& exec [[maybe_unused]], ForwardIterator first, ForwardIterator last, const T& value)
{
  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<ForwardIterator, ::hpx::forward_traversal_tag>)
  {
    return ::hpx::fill(hpx::detail::to_hpx_execution_policy(exec),
                       ::thrust::try_unwrap_contiguous_iterator(first),
                       ::thrust::try_unwrap_contiguous_iterator(last),
                       value);
  }
  else
  {
    return ::hpx::fill(first, last, value);
  }
}

template <typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
OutputIterator
fill_n(execution_policy<DerivedPolicy>& exec [[maybe_unused]], OutputIterator first, Size n, const T& value)
{
  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<OutputIterator, ::hpx::forward_traversal_tag>)
  {
    auto res = ::hpx::fill_n(
      hpx::detail::to_hpx_execution_policy(exec), ::thrust::try_unwrap_contiguous_iterator(first), n, value);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::fill_n(first, n, value);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file equal.h
 *  \brief HPX implementation of equal.
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

#include <hpx/parallel/algorithms/equal.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
bool equal(execution_policy<DerivedPolicy>& exec [[maybe_unused]],
           InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2)
{
  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator1, ::hpx::forward_traversal_tag>
                && ::hpx::traits::belongs_to_iterator_traversal_v<InputIterator2, ::hpx::forward_traversal_tag>)
  {
    return ::hpx::equal(hpx::detail::to_hpx_execution_policy(exec),
                        ::thrust::try_unwrap_contiguous_iterator(first1),
                        ::thrust::try_unwrap_contiguous_iterator(last1),
                        ::thrust::try_unwrap_contiguous_iterator(first2));
  }
  else
  {
    return ::hpx::equal(first1, last1, first2);
  }
}

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
bool equal(execution_policy<DerivedPolicy>& exec [[maybe_unused]],
           InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2,
           BinaryPredicate binary_pred)
{
  // wrap pred
  hpx_wrapped_function<BinaryPredicate> wrapped_binary_pred{binary_pred};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator1, ::hpx::forward_traversal_tag>
                && ::hpx::traits::belongs_to_iterator_traversal_v<InputIterator2, ::hpx::forward_traversal_tag>)
  {
    return ::hpx::equal(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      wrapped_binary_pred);
  }
  else
  {
    return ::hpx::equal(first1, last1, first2, wrapped_binary_pred);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END

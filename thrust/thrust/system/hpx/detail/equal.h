/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
bool equal(execution_policy<DerivedPolicy>& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>)
  {
      return ::hpx::equal(hpx::detail::to_hpx_execution_policy(exec),
                          detail::try_unwrap_contiguous_iterator(first1),
                          detail::try_unwrap_contiguous_iterator(last1),
                          detail::try_unwrap_contiguous_iterator(first2));
  }
  else
  {
    (void) exec;
    return ::hpx::equal(first1, last1, first2);
  }
}

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
bool equal(execution_policy<DerivedPolicy>& exec,
           InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2,
           BinaryPredicate binary_pred)
{
  // wrap pred
  wrapped_function<BinaryPredicate> wrapped_binary_pred{binary_pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>)
  {
      return ::hpx::equal(
        hpx::detail::to_hpx_execution_policy(exec),
        detail::try_unwrap_contiguous_iterator(first1),
        detail::try_unwrap_contiguous_iterator(last1),
        detail::try_unwrap_contiguous_iterator(first2),
        wrapped_binary_pred);
  }
  else
  {
    (void) exec;
    return ::hpx::equal(first1, last1, first2, wrapped_binary_pred);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

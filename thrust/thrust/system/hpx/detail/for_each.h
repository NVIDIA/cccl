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
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator, typename UnaryFunction>
InputIterator for_each(execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, UnaryFunction f)
{
  // wrap f
  wrapped_function<UnaryFunction> wrapped_f{f};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
      (void) ::hpx::for_each(
        hpx::detail::to_hpx_execution_policy(exec),
        ::thrust::try_unwrap_contiguous_iterator(first),
        ::thrust::try_unwrap_contiguous_iterator(last),
        wrapped_f);
  }
  else
  {
    (void) exec;
    (void) ::hpx::for_each(first, last, wrapped_f);
  }

  return last;
}

template <typename DerivedPolicy, typename InputIterator, typename Size, typename UnaryFunction>
InputIterator for_each_n(execution_policy<DerivedPolicy>& exec, InputIterator first, Size n, UnaryFunction f)
{
  // wrap f
  wrapped_function<UnaryFunction> wrapped_f{f};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
      auto res = ::hpx::for_each_n(
        hpx::detail::to_hpx_execution_policy(exec), ::thrust::try_unwrap_contiguous_iterator(first), n, wrapped_f);
      return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    (void) exec;
    return ::hpx::for_each_n(first, n, wrapped_f);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

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
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator transform(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction op)
{
  // wrap op
  wrapped_function<UnaryFunction> wrapped_op{op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
      auto res = ::hpx::transform(
        hpx::detail::to_hpx_execution_policy(exec),
        detail::try_unwrap_contiguous_iterator(first),
        detail::try_unwrap_contiguous_iterator(last),
        detail::try_unwrap_contiguous_iterator(result),
        wrapped_op);
      return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    (void) exec;
    return ::hpx::transform(first, last, result, wrapped_op);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryFunction>
OutputIterator transform(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryFunction op)
{
  // wrap op
  wrapped_function<BinaryFunction> wrapped_op{op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>)
  {
      auto res = ::hpx::transform(
        hpx::detail::to_hpx_execution_policy(exec),
        detail::try_unwrap_contiguous_iterator(first1),
        detail::try_unwrap_contiguous_iterator(last1),
        detail::try_unwrap_contiguous_iterator(first2),
        detail::try_unwrap_contiguous_iterator(result),
        wrapped_op);
      return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    (void) exec;
    return ::hpx::transform(first1, last1, first2, result, wrapped_op);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

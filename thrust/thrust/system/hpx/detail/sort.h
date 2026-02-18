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
#include <hpx/parallel/algorithms/stable_sort.hpp>


THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
void stable_sort(
  execution_policy<DerivedPolicy>& exec, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
{
  // wrap comp
  wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_random_access_iterator_v<RandomAccessIterator>)
  {
      return ::hpx::stable_sort(
        hpx::detail::to_hpx_execution_policy(exec),
        ::thrust::try_unwrap_contiguous_iterator(first),
        ::thrust::try_unwrap_contiguous_iterator(last),
        wrapped_comp);
  }
  else
  {
    (void) exec;
    return ::hpx::stable_sort(first, last, comp);
  }
}
} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

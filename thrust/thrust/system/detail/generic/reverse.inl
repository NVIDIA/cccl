/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/swap.h>
#include <thrust/system/detail/generic/reverse.h>

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/reverse_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename ExecutionPolicy, typename BidirectionalIterator>
_CCCL_HOST_DEVICE void
reverse(thrust::execution_policy<ExecutionPolicy>& exec, BidirectionalIterator first, BidirectionalIterator last)
{
  using difference_type = thrust::detail::it_difference_t<BidirectionalIterator>;

  // find the midpoint of [first,last)
  difference_type N = ::cuda::std::distance(first, last);
  BidirectionalIterator mid(first);
  ::cuda::std::advance(mid, N / 2);

  // swap elements of [first,mid) with [last - 1, mid)
  thrust::swap_ranges(exec, first, mid, ::cuda::std::make_reverse_iterator(last));
} // end reverse()

template <typename ExecutionPolicy, typename BidirectionalIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator reverse_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  BidirectionalIterator first,
  BidirectionalIterator last,
  OutputIterator result)
{
  return thrust::copy(exec, ::cuda::std::reverse_iterator{last}, ::cuda::std::reverse_iterator{first}, result);
} // end reverse_copy()
} // namespace system::detail::generic
THRUST_NAMESPACE_END

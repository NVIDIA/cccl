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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reverse.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/reverse.h>
#include <thrust/system/detail/sequential/reverse.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(reverse.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(reverse.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/reverse.h>
#  include <thrust/system/cuda/detail/reverse.h>
#  include <thrust/system/omp/detail/reverse.h>
#  include <thrust/system/tbb/detail/reverse.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename BidirectionalIterator>
_CCCL_HOST_DEVICE void reverse(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                               BidirectionalIterator first,
                               BidirectionalIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reverse");
  using thrust::system::detail::generic::reverse;
  return reverse(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end reverse()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename BidirectionalIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator reverse_copy(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  BidirectionalIterator first,
  BidirectionalIterator last,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reverse_copy");
  using thrust::system::detail::generic::reverse_copy;
  return reverse_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end reverse_copy()

template <typename BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reverse");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<BidirectionalIterator>::type;

  System system;

  return thrust::reverse(select_system(system), first, last);
} // end reverse()

template <typename BidirectionalIterator, typename OutputIterator>
OutputIterator reverse_copy(BidirectionalIterator first, BidirectionalIterator last, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reverse_copy");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<BidirectionalIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::reverse_copy(select_system(system1, system2), first, last, result);
} // end reverse_copy()

THRUST_NAMESPACE_END

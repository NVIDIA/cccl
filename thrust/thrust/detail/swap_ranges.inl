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
#include <thrust/swap.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/swap_ranges.h>
#include <thrust/system/detail/sequential/swap_ranges.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(swap_ranges.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(swap_ranges.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/swap_ranges.h>
#  include <thrust/system/cuda/detail/swap_ranges.h>
#  include <thrust/system/omp/detail/swap_ranges.h>
#  include <thrust/system/tbb/detail/swap_ranges.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator2 swap_ranges(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::swap_ranges");
  using thrust::system::detail::generic::swap_ranges;
  return swap_ranges(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2);
} // end swap_ranges()

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator2 swap_ranges(ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::swap_ranges");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<ForwardIterator1>::type;
  using System2 = typename thrust::iterator_system<ForwardIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::swap_ranges(select_system(system1, system2), first1, last1, first2);
} // end swap_ranges()

THRUST_NAMESPACE_END

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
#include <thrust/count.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/count.h>
#include <thrust/system/detail/sequential/count.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(count.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(count.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/count.h>
#  include <thrust/system/cuda/detail/count.h>
#  include <thrust/system/omp/detail/count.h>
#  include <thrust/system/tbb/detail/count.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename EqualityComparable>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator>
count(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
      InputIterator first,
      InputIterator last,
      const EqualityComparable& value)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::count");
  using thrust::system::detail::generic::count;
  return count(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, value);
} // end count()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<InputIterator>
count_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         InputIterator first,
         InputIterator last,
         Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::count_if");
  using thrust::system::detail::generic::count_if;
  return count_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end count_if()

template <typename InputIterator, typename EqualityComparable>
thrust::detail::it_difference_t<InputIterator>
count(InputIterator first, InputIterator last, const EqualityComparable& value)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::count");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::count(select_system(system), first, last, value);
} // end count()

template <typename InputIterator, typename Predicate>
thrust::detail::it_difference_t<InputIterator> count_if(InputIterator first, InputIterator last, Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::count_if");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::count_if(select_system(system), first, last, pred);
} // end count_if()

THRUST_NAMESPACE_END

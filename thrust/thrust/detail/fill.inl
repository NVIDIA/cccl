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

#include <thrust/detail/nvtx_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/fill.h>
#include <thrust/system/detail/sequential/fill.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(fill.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(fill.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/fill.h>
#  include <thrust/system/cuda/detail/fill.h>
#  include <thrust/system/omp/detail/fill.h>
#  include <thrust/system/tbb/detail/fill.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void
fill(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
     ForwardIterator first,
     ForwardIterator last,
     const T& value)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<DerivedPolicy>(), "thrust::fill");
  using thrust::system::detail::generic::fill;
  return fill(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, value);
} // end fill()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
_CCCL_HOST_DEVICE OutputIterator
fill_n(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, OutputIterator first, Size n, const T& value)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<DerivedPolicy>(), "thrust::fill_n");
  using thrust::system::detail::generic::fill_n;
  return fill_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, value);
} // end fill_n()

template <typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void fill(ForwardIterator first, ForwardIterator last, const T& value)
{
  using System = typename thrust::iterator_system<ForwardIterator>::type;
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<System>(), "thrust::fill");
  using thrust::system::detail::generic::select_system;

  System system;

  thrust::fill(select_system(system), first, last, value);
} // end fill()

template <typename OutputIterator, typename Size, typename T>
_CCCL_HOST_DEVICE OutputIterator fill_n(OutputIterator first, Size n, const T& value)
{
  using System = typename thrust::iterator_system<OutputIterator>::type;
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<System>(), "thrust::fill_n");
  using thrust::system::detail::generic::select_system;

  System system;

  return thrust::fill_n(select_system(system), first, n, value);
} // end fill()

THRUST_NAMESPACE_END

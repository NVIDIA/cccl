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
#include <thrust/extrema.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/extrema.h>
#include <thrust/system/detail/sequential/extrema.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(extrema.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(extrema.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/extrema.h>
#  include <thrust/system/cuda/detail/extrema.h>
#  include <thrust/system/omp/detail/extrema.h>
#  include <thrust/system/tbb/detail/extrema.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator min_element(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("min_element");
  using thrust::system::detail::generic::min_element;
  return min_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end min_element()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator min_element(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate comp)
{
  _CCCL_NVTX_RANGE_SCOPE("min_element");
  using thrust::system::detail::generic::min_element;
  return min_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end min_element()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ForwardIterator max_element(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("max_element");
  using thrust::system::detail::generic::max_element;
  return max_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end max_element()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator max_element(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate comp)
{
  _CCCL_NVTX_RANGE_SCOPE("max_element");
  using thrust::system::detail::generic::max_element;
  return max_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end max_element()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("minmax_element");
  using thrust::system::detail::generic::minmax_element;
  return minmax_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end minmax_element()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate comp)
{
  _CCCL_NVTX_RANGE_SCOPE("minmax_element");
  using thrust::system::detail::generic::minmax_element;
  return minmax_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end minmax_element()

template <typename ForwardIterator>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("min_element");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::min_element(select_system(system), first, last);
} // end min_element()

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  _CCCL_NVTX_RANGE_SCOPE("min_element");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::min_element(select_system(system), first, last, comp);
} // end min_element()

template <typename ForwardIterator>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("max_element");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::max_element(select_system(system), first, last);
} // end max_element()

template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  _CCCL_NVTX_RANGE_SCOPE("max_element");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::max_element(select_system(system), first, last, comp);
} // end max_element()

template <typename ForwardIterator>
::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(ForwardIterator first, ForwardIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("minmax_element");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::minmax_element(select_system(system), first, last);
} // end minmax_element()

template <typename ForwardIterator, typename BinaryPredicate>
::cuda::std::pair<ForwardIterator, ForwardIterator>
minmax_element(ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  _CCCL_NVTX_RANGE_SCOPE("minmax_element");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::minmax_element(select_system(system), first, last, comp);
} // end minmax_element()

THRUST_NAMESPACE_END

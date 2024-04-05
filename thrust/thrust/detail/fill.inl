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

#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/fill.h>
#include <thrust/system/detail/generic/fill.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void
fill(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
     ForwardIterator first,
     ForwardIterator last,
     const T& value)
{
  using thrust::system::detail::generic::fill;
  return fill(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, value);
} // end fill()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
_CCCL_HOST_DEVICE OutputIterator
fill_n(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, OutputIterator first, Size n, const T& value)
{
  using thrust::system::detail::generic::fill_n;
  return fill_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, value);
} // end fill_n()

template <typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void fill(ForwardIterator first, ForwardIterator last, const T& value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  thrust::fill(select_system(system), first, last, value);
} // end fill()

template <typename OutputIterator, typename Size, typename T>
_CCCL_HOST_DEVICE OutputIterator fill_n(OutputIterator first, Size n, const T& value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<OutputIterator>::type System;

  System system;

  return thrust::fill_n(select_system(system), first, n, value);
} // end fill()

THRUST_NAMESPACE_END

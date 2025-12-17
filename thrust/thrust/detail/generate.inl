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
#include <thrust/generate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/generate.h>
#include <thrust/system/detail/sequential/generate.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(generate.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(generate.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/generate.h>
#  include <thrust/system/cuda/detail/generate.h>
#  include <thrust/system/omp/detail/generate.h>
#  include <thrust/system/tbb/detail/generate.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename Generator>
_CCCL_HOST_DEVICE void
generate(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         ForwardIterator first,
         ForwardIterator last,
         Generator gen)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<DerivedPolicy>(), "thrust::generate");
  using thrust::system::detail::generic::generate;
  return generate(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, gen);
} // end generate()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename OutputIterator, typename Size, typename Generator>
_CCCL_HOST_DEVICE OutputIterator generate_n(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, OutputIterator first, Size n, Generator gen)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<DerivedPolicy>(), "thrust::generate_n");
  using thrust::system::detail::generic::generate_n;
  return generate_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, gen);
} // end generate_n()

template <typename ForwardIterator, typename Generator>
void generate(ForwardIterator first, ForwardIterator last, Generator gen)
{
  using System = typename thrust::iterator_system<ForwardIterator>::type;
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<System>(), "thrust::generate");
  using thrust::system::detail::generic::select_system;

  System system;

  return thrust::generate(select_system(system), first, last, gen);
} // end generate()

template <typename OutputIterator, typename Size, typename Generator>
OutputIterator generate_n(OutputIterator first, Size n, Generator gen)
{
  using System = typename thrust::iterator_system<OutputIterator>::type;
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<System>(), "thrust::generate_n");
  using thrust::system::detail::generic::select_system;

  System system;

  return thrust::generate_n(select_system(system), first, n, gen);
} // end generate_n()

THRUST_NAMESPACE_END

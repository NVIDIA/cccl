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
#include <thrust/scatter.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/scatter.h>
#include <thrust/system/detail/sequential/scatter.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(scatter.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(scatter.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/scatter.h>
#  include <thrust/system/cuda/detail/scatter.h>
#  include <thrust/system/omp/detail/scatter.h>
#  include <thrust/system/tbb/detail/scatter.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
scatter(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
        InputIterator1 first,
        InputIterator1 last,
        InputIterator2 map,
        RandomAccessIterator output)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::scatter");
  using thrust::system::detail::generic::scatter;
  return scatter(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, map, output);
} // end scatter()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator>
_CCCL_HOST_DEVICE void scatter_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::scatter_if");
  using thrust::system::detail::generic::scatter_if;
  return scatter_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, map, stencil, output);
} // end scatter_if()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator,
          typename Predicate>
_CCCL_HOST_DEVICE void scatter_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output,
  Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::scatter_if");
  using thrust::system::detail::generic::scatter_if;
  return scatter_if(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, map, stencil, output, pred);
} // end scatter_if()

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(InputIterator1 first, InputIterator1 last, InputIterator2 map, RandomAccessIterator output)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::scatter");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<RandomAccessIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::scatter(select_system(system1, system2, system3), first, last, map, output);
} // end scatter()

template <typename InputIterator1, typename InputIterator2, typename InputIterator3, typename RandomAccessIterator>
void scatter_if(
  InputIterator1 first, InputIterator1 last, InputIterator2 map, InputIterator3 stencil, RandomAccessIterator output)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::scatter_if");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<RandomAccessIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::scatter_if(select_system(system1, system2, system3, system4), first, last, map, stencil, output);
} // end scatter_if()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator,
          typename Predicate>
void scatter_if(InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output,
                Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::scatter_if");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<RandomAccessIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::scatter_if(select_system(system1, system2, system3, system4), first, last, map, stencil, output, pred);
} // end scatter_if()

THRUST_NAMESPACE_END

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
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/set_operations.h>
#include <thrust/system/detail/sequential/set_operations.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(set_operations.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(set_operations.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/set_operations.h>
#  include <thrust/system/cuda/detail/set_operations.h>
#  include <thrust/system/omp/detail/set_operations.h>
#  include <thrust/system/tbb/detail/set_operations.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference");
  using thrust::system::detail::generic::set_difference;
  return set_difference(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_difference()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE OutputIterator set_difference(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference");
  using thrust::system::detail::generic::set_difference;
  return set_difference(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_difference()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference_by_key");
  using thrust::system::detail::generic::set_difference_by_key;
  return set_difference_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result);
} // end set_difference_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference_by_key");
  using thrust::system::detail::generic::set_difference_by_key;
  return set_difference_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
} // end set_difference_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection");
  using thrust::system::detail::generic::set_intersection;
  return set_intersection(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_intersection()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE OutputIterator set_intersection(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection");
  using thrust::system::detail::generic::set_intersection;
  return set_intersection(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_intersection()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection_by_key");
  using thrust::system::detail::generic::set_intersection_by_key;
  return set_intersection_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    keys_result,
    values_result);
} // end set_intersection_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection_by_key");
  using thrust::system::detail::generic::set_intersection_by_key;
  return set_intersection_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    keys_result,
    values_result,
    comp);
} // end set_intersection_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference");
  using thrust::system::detail::generic::set_symmetric_difference;
  return set_symmetric_difference(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_symmetric_difference()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE OutputIterator set_symmetric_difference(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference");
  using thrust::system::detail::generic::set_symmetric_difference;
  return set_symmetric_difference(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_symmetric_difference()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference_by_key");
  using thrust::system::detail::generic::set_symmetric_difference_by_key;
  return set_symmetric_difference_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result);
} // end set_symmetric_difference_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference_by_key");
  using thrust::system::detail::generic::set_symmetric_difference_by_key;
  return set_symmetric_difference_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
} // end set_symmetric_difference_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator set_union(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union");
  using thrust::system::detail::generic::set_union;
  return set_union(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result);
} // end set_union()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE OutputIterator set_union(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union");
  using thrust::system::detail::generic::set_union;
  return set_union(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, last2, result, comp);
} // end set_union()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union_by_key");
  using thrust::system::detail::generic::set_union_by_key;
  return set_union_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result);
} // end set_union_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakCompare>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakCompare comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union_by_key");
  using thrust::system::detail::generic::set_union_by_key;
  return set_union_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
} // end set_union_by_key()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator set_difference(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_difference(select_system(system1, system2, system3), first1, last1, first2, last2, result, comp);
} // end set_difference()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_difference(
  InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_difference(select_system(system1, system2, system3), first1, last1, first2, last2, result);
} // end set_difference()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<InputIterator4>::type;
  using System5 = typename thrust::iterator_system<OutputIterator1>::type;
  using System6 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_difference_by_key(
    select_system(system1, system2, system3, system4, system5, system6),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
} // end set_difference_by_key()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_difference_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<InputIterator4>::type;
  using System5 = typename thrust::iterator_system<OutputIterator1>::type;
  using System6 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_difference_by_key(
    select_system(system1, system2, system3, system4, system5, system6),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result);
} // end set_difference_by_key()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator set_intersection(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_intersection(select_system(system1, system2, system3), first1, last1, first2, last2, result, comp);
} // end set_intersection()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_intersection(
  InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_intersection(select_system(system1, system2, system3), first1, last1, first2, last2, result);
} // end set_intersection()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<OutputIterator1>::type;
  using System5 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;

  return thrust::set_intersection_by_key(
    select_system(system1, system2, system3, system4, system5),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    keys_result,
    values_result,
    comp);
} // end set_intersection_by_key()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_intersection_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<OutputIterator1>::type;
  using System5 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;

  return thrust::set_intersection_by_key(
    select_system(system1, system2, system3, system4, system5),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    keys_result,
    values_result);
} // end set_intersection_by_key()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator set_symmetric_difference(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_symmetric_difference(
    select_system(system1, system2, system3), first1, last1, first2, last2, result, comp);
} // end set_symmetric_difference()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_symmetric_difference(
  InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_symmetric_difference(
    select_system(system1, system2, system3), first1, last1, first2, last2, result);
} // end set_symmetric_difference()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<InputIterator4>::type;
  using System5 = typename thrust::iterator_system<OutputIterator1>::type;
  using System6 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_symmetric_difference_by_key(
    select_system(system1, system2, system3, system4, system5, system6),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
} // end set_symmetric_difference_by_key()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_symmetric_difference_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<InputIterator4>::type;
  using System5 = typename thrust::iterator_system<OutputIterator1>::type;
  using System6 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_symmetric_difference_by_key(
    select_system(system1, system2, system3, system4, system5, system6),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result);
} // end set_symmetric_difference_by_key()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename StrictWeakOrdering>
OutputIterator set_union(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_union(select_system(system1, system2, system3), first1, last1, first2, last2, result, comp);
} // end set_union()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_union(
  InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::set_union(select_system(system1, system2, system3), first1, last1, first2, last2, result);
} // end set_union()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<InputIterator4>::type;
  using System5 = typename thrust::iterator_system<OutputIterator1>::type;
  using System6 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_union_by_key(
    select_system(system1, system2, system3, system4, system5, system6),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
} // end set_union_by_key()

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::set_union_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<InputIterator3>::type;
  using System4 = typename thrust::iterator_system<InputIterator4>::type;
  using System5 = typename thrust::iterator_system<OutputIterator1>::type;
  using System6 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;
  System5 system5;
  System6 system6;

  return thrust::set_union_by_key(
    select_system(system1, system2, system3, system4, system5, system6),
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result);
} // end set_union_by_key()

THRUST_NAMESPACE_END

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
#include <thrust/scan.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/transform_scan.h>
#include <thrust/system/detail/sequential/transform_scan.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(transform_scan.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(transform_scan.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/transform_scan.h>
#  include <thrust/system/cuda/detail/transform_scan.h>
#  include <thrust/system/omp/detail/transform_scan.h>
#  include <thrust/system/tbb/detail/transform_scan.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator transform_inclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_inclusive_scan");
  using thrust::system::detail::generic::transform_inclusive_scan;
  return transform_inclusive_scan(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, unary_op, binary_op);
} // end transform_inclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator transform_inclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_inclusive_scan");
  using thrust::system::detail::generic::transform_inclusive_scan;
  return transform_inclusive_scan(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, unary_op, init, binary_op);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator transform_exclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_exclusive_scan");
  using thrust::system::detail::generic::transform_exclusive_scan;
  return transform_exclusive_scan(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, unary_op, init, binary_op);
} // end transform_exclusive_scan()

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename BinaryFunction>
OutputIterator transform_inclusive_scan(
  InputIterator first, InputIterator last, OutputIterator result, UnaryFunction unary_op, BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_inclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::transform_inclusive_scan(select_system(system1, system2), first, last, result, unary_op, binary_op);
} // end transform_inclusive_scan()

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
OutputIterator transform_inclusive_scan(
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_inclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::transform_inclusive_scan(
    select_system(system1, system2), first, last, result, unary_op, init, binary_op);
} // end transform_inclusive_scan()

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
OutputIterator transform_exclusive_scan(
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::transform_exclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::transform_exclusive_scan(
    select_system(system1, system2), first, last, result, unary_op, init, binary_op);
} // end transform_exclusive_scan()

THRUST_NAMESPACE_END

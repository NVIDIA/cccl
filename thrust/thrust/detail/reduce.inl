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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reduce.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/reduce.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/system/detail/sequential/reduce.h>
#include <thrust/system/detail/sequential/reduce_by_key.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(reduce.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(reduce.h)
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(reduce_by_key.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(reduce_by_key.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/reduce.h>
#  include <thrust/system/cpp/detail/reduce_by_key.h>
#  include <thrust/system/cuda/detail/reduce.h>
#  include <thrust/system/cuda/detail/reduce_by_key.h>
#  include <thrust/system/omp/detail/reduce.h>
#  include <thrust/system/omp/detail/reduce_by_key.h>
#  include <thrust/system/tbb/detail/reduce.h>
#  include <thrust/system/tbb/detail/reduce_by_key.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator>
_CCCL_HOST_DEVICE detail::it_value_t<InputIterator>
reduce(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, InputIterator first, InputIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE_IF(detail::should_enable_nvtx_for_policy<DerivedPolicy>(), "thrust::reduce");
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end reduce()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename T>
_CCCL_HOST_DEVICE T reduce(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, InputIterator first, InputIterator last, T init)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce");
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, init);
} // end reduce()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename T, typename BinaryFunction>
_CCCL_HOST_DEVICE T reduce(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  T init,
  BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce");
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, init, binary_op);
} // end reduce()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE void reduce_into(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output)
{
  using thrust::system::detail::generic::reduce_into;
  reduce_into(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, output);
} // end reduce_into()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE void reduce_into(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  T init)
{
  using thrust::system::detail::generic::reduce_into;
  reduce_into(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, output, init);
} // end reduce_into()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
_CCCL_HOST_DEVICE void reduce_into(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  T init,
  BinaryFunction binary_op)
{
  using thrust::system::detail::generic::reduce_into;
  reduce_into(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, output, init, binary_op);
} // end reduce_into()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce_by_key");
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first,
    keys_last,
    values_first,
    keys_output,
    values_output);
} // end reduce_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce_by_key");
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first,
    keys_last,
    values_first,
    keys_output,
    values_output,
    binary_pred);
} // end reduce_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce_by_key");
  using thrust::system::detail::generic::reduce_by_key;
  return reduce_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    keys_first,
    keys_last,
    values_first,
    keys_output,
    values_output,
    binary_pred,
    binary_op);
} // end reduce_by_key()

template <typename InputIterator>
detail::it_value_t<InputIterator> reduce(InputIterator first, InputIterator last)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::reduce(select_system(system), first, last);
}

template <typename InputIterator, typename T>
T reduce(InputIterator first, InputIterator last, T init)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::reduce(select_system(system), first, last, init);
}

template <typename InputIterator, typename T, typename BinaryFunction>
T reduce(InputIterator first, InputIterator last, T init, BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<InputIterator>::type;

  System system;

  return thrust::reduce(select_system(system), first, last, init, binary_op);
}

template <typename InputIterator, typename OutputIterator>
void reduce_into(InputIterator first, InputIterator last, OutputIterator output)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  thrust::reduce_into(select_system(system1, system2), first, last, output);
}

template <typename InputIterator, typename OutputIterator, typename T>
void reduce_into(InputIterator first, InputIterator last, OutputIterator output, T init)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  thrust::reduce_into(select_system(system1, system2), first, last, output, init);
}

template <typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
void reduce_into(InputIterator first, InputIterator last, OutputIterator output, T init, BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  thrust::reduce_into(select_system(system1, system2), first, last, output, init, binary_op);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator1, typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator1>::type;
  using System4 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::reduce_by_key(
    select_system(system1, system2, system3, system4), keys_first, keys_last, values_first, keys_output, values_output);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator1>::type;
  using System4 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::reduce_by_key(
    select_system(system1, system2, system3, system4),
    keys_first,
    keys_last,
    values_first,
    keys_output,
    values_output,
    binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::reduce_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator1>::type;
  using System4 = typename thrust::iterator_system<OutputIterator2>::type;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::reduce_by_key(
    select_system(system1, system2, system3, system4),
    keys_first,
    keys_last,
    values_first,
    keys_output,
    values_output,
    binary_pred,
    binary_op);
}

THRUST_NAMESPACE_END

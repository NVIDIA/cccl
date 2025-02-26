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
#include <thrust/detail/copy_if.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/remove.h>
#include <thrust/system/detail/generic/remove.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE ForwardIterator
remove(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, const T& value)
{
  thrust::detail::equal_to_value<T> pred(value);

  // XXX consider using a placeholder here
  return thrust::remove_if(exec, first, last, pred);
} // end remove()

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator remove_copy(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  const T& value)
{
  thrust::detail::equal_to_value<T> pred(value);

  // XXX consider using a placeholder here
  return thrust::remove_copy_if(exec, first, last, result, pred);
} // end remove_copy()

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator
remove_if(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  // create temporary storage for an intermediate result
  thrust::detail::temporary_array<InputType, DerivedPolicy> temp(exec, first, last);

  // remove into temp
  return thrust::remove_copy_if(exec, temp.begin(), temp.end(), temp.begin(), first, pred);
} // end remove_if()

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
_CCCL_HOST_DEVICE ForwardIterator remove_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;

  // create temporary storage for an intermediate result
  thrust::detail::temporary_array<InputType, DerivedPolicy> temp(exec, first, last);

  // remove into temp
  return thrust::remove_copy_if(exec, temp.begin(), temp.end(), stencil, first, pred);
} // end remove_if()

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
_CCCL_HOST_DEVICE OutputIterator remove_copy_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred)
{
  return thrust::remove_copy_if(exec, first, last, first, result, pred);
} // end remove_copy_if()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
_CCCL_HOST_DEVICE OutputIterator remove_copy_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred)
{
  return thrust::copy_if(exec, first, last, stencil, result, thrust::not_fn(pred));
} // end remove_copy_if()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

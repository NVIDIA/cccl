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

_CCCL_IMPLICIT_SYSTEM_HEADER
#include <thrust/detail/internal_functional.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result)
{
  using T = typename thrust::iterator_value<InputIterator>::type;
  return thrust::transform(exec, first, last, result, thrust::identity<T>());
} // end copy()

template <typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy_n(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, Size n, OutputIterator result)
{
  using value_type = typename thrust::iterator_value<InputIterator>::type;
  using xfrm_type  = thrust::identity<value_type>;

  using functor_type = thrust::detail::unary_transform_functor<xfrm_type>;

  using iterator_tuple = thrust::tuple<InputIterator, OutputIterator>;
  using zip_iter       = thrust::zip_iterator<iterator_tuple>;

  zip_iter zipped = thrust::make_zip_iterator(thrust::make_tuple(first, result));

  return thrust::get<1>(thrust::for_each_n(exec, zipped, n, functor_type(xfrm_type())).get_iterator_tuple());
} // end copy_n()

} // namespace generic
} // namespace detail
} // namespace system
THRUST_NAMESPACE_END

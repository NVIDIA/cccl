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
#include <thrust/detail/internal_functional.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result)
{
  return thrust::transform(exec, first, last, result, ::cuda::std::identity{});
} // end copy()

template <typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator
copy_n(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, Size n, OutputIterator result)
{
  using xfrm_type = ::cuda::std::identity;

  using functor_type = thrust::detail::unary_transform_functor<xfrm_type>;

  using iterator_tuple = thrust::tuple<InputIterator, OutputIterator>;
  using zip_iter       = thrust::zip_iterator<iterator_tuple>;

  zip_iter zipped = thrust::make_zip_iterator(first, result);

  return thrust::get<1>(thrust::for_each_n(exec, zipped, n, functor_type{xfrm_type()}).get_iterator_tuple());
} // end copy_n()
} // namespace system::detail::generic
THRUST_NAMESPACE_END

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

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename, typename>
class permutation_iterator;

namespace detail
{

template <typename ElementIterator, typename IndexIterator>
struct permutation_iterator_base
{
  using System1 = typename thrust::iterator_system<ElementIterator>::type;
  using System2 = typename thrust::iterator_system<IndexIterator>::type;

  using type =
    thrust::iterator_adaptor<permutation_iterator<ElementIterator, IndexIterator>,
                             IndexIterator,
                             typename thrust::iterator_value<ElementIterator>::type,
                             typename detail::minimum_system<System1, System2>::type,
                             thrust::use_default,
                             typename thrust::iterator_reference<ElementIterator>::type>;
}; // end permutation_iterator_base

} // namespace detail

THRUST_NAMESPACE_END

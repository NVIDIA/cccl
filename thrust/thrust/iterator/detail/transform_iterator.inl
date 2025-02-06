/*
 *  Copyright 2008-2021 NVIDIA Corporation
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
#include <thrust/detail/functional/actor.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

template <class UnaryFunction, class Iterator, class Reference, class Value>
class transform_iterator;

namespace detail
{

template <class UnaryFunc, class Iterator>
struct transform_iterator_reference
{
  // by default, dereferencing the iterator yields the same as the function.
  using type = decltype(::cuda::std::declval<UnaryFunc>()(::cuda::std::declval<iterator_value_t<Iterator>>()));
};

// for certain function objects, we need to tweak the reference type. Notably, identity functions must decay to values.
// See the implementation of transform_iterator<...>::dereference() for several comments on why this is necessary.
_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename T, class Iterator>
struct transform_iterator_reference<identity<T>, Iterator>
{
  using type = T;
};
template <class Iterator>
struct transform_iterator_reference<identity<>, Iterator>
{
  using type = iterator_value_t<Iterator>;
};
_CCCL_SUPPRESS_DEPRECATED_POP
template <class Iterator>
struct transform_iterator_reference<::cuda::std::identity, Iterator>
{
  using type = iterator_value_t<Iterator>;
};
template <typename Eval, class Iterator>
struct transform_iterator_reference<functional::actor<Eval>, Iterator>
{
  using type = ::cuda::std::remove_reference_t<decltype(::cuda::std::declval<functional::actor<Eval>>()(
    ::cuda::std::declval<iterator_value_t<Iterator>>()))>;
};

// Type function to compute the iterator_adaptor instantiation to be used for transform_iterator
template <class UnaryFunc, class Iterator, class Reference, class Value>
struct make_transform_iterator_base
{
private:
  using reference  = typename ia_dflt_help<Reference, transform_iterator_reference<UnaryFunc, Iterator>>::type;
  using value_type = typename ia_dflt_help<Value, ::cuda::std::remove_cvref<reference>>::type;

public:
  using type =
    iterator_adaptor<transform_iterator<UnaryFunc, Iterator, Reference, Value>,
                     Iterator,
                     value_type,
                     use_default,
                     typename iterator_traits<Iterator>::iterator_category,
                     reference>;
};

} // namespace detail
THRUST_NAMESPACE_END

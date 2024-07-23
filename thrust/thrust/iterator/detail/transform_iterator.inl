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
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/result_of_adaptable_function.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/type_traits/remove_cvref.h>

THRUST_NAMESPACE_BEGIN

template <class UnaryFunction, class Iterator, class Reference, class Value>
class transform_iterator;

namespace detail
{

// Compute the iterator_adaptor instantiation to be used for transform_iterator
template <class UnaryFunc, class Iterator, class Reference, class Value>
struct transform_iterator_base
{
private:
  using unary_func_input_t =
    ::cuda::std::_If<::cuda::std::is_lvalue_reference<iterator_reference_t<Iterator>>::value,
                     iterator_reference_t<Iterator>,
                     const iterator_value_t<Iterator>&>;

  // By default, dereferencing the iterator yields the same as the function.
  using reference = typename thrust::detail::
    ia_dflt_help<Reference, thrust::detail::result_of_adaptable_function<UnaryFunc(unary_func_input_t)>>::type;

  // To get the default for Value: remove cvref on the result type.
  using value_type = typename thrust::detail::ia_dflt_help<Value, thrust::remove_cvref<reference>>::type;

public:
  using type =
    thrust::iterator_adaptor<transform_iterator<UnaryFunc, Iterator, Reference, Value>,
                             Iterator,
                             value_type,
                             thrust::use_default,
                             typename thrust::iterator_traits<Iterator>::iterator_category,
                             reference>;
};

} // namespace detail
THRUST_NAMESPACE_END

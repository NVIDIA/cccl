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
#include <thrust/detail/type_traits/result_of_adaptable_function.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/type_traits/remove_cvref.h>

THRUST_NAMESPACE_BEGIN

template <class UnaryFunction, class Iterator, class Reference, class Value>
class transform_iterator;

namespace detail
{

// Type function to compute the iterator_adaptor instantiation to be used for transform_iterator
template <class UnaryFunc, class Iterator, class Reference, class Value>
struct make_transform_iterator_base
{
  // We need to decay any proxy reference type that cannot be decayed to a raw reference (e.g.
  // std::vector<bool>::reference). Thrust's wrapped references can be decayed.
  static constexpr bool base_iter_ref_needs_decay =
    !::cuda::std::is_same<iterator_value_t<Iterator>, ::cuda::std::__decay_t<iterator_reference_t<Iterator>>>::value
    && !is_wrapped_reference<iterator_reference_t<Iterator>>::value;

  using func_input_t =
    ::cuda::std::_If<base_iter_ref_needs_decay,
                     const iterator_value_t<Iterator>&, // decay the reference to the value type
                     typename raw_reference<iterator_reference_t<Iterator>>::type>;
  using func_return_t = ::cuda::std::__invoke_of<UnaryFunc, func_input_t>;

  // By default, dereferencing the iterator yields the same as the function.
  using reference  = typename ia_dflt_help<Reference, func_return_t>::type;
  using value_type = typename ia_dflt_help<Value, remove_cvref<reference>>::type;

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

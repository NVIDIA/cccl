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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/iterator_categories.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename Category>
using host_system_category_to_traversal = ::cuda::std::_If<
  ::cuda::std::is_convertible_v<Category, random_access_host_iterator_tag>,
  random_access_traversal_tag,
  ::cuda::std::_If<
    ::cuda::std::is_convertible_v<Category, bidirectional_host_iterator_tag>,
    bidirectional_traversal_tag,
    ::cuda::std::_If<::cuda::std::is_convertible_v<Category, forward_host_iterator_tag>,
                     forward_traversal_tag,
                     ::cuda::std::_If<::cuda::std::is_convertible_v<Category, input_host_iterator_tag>,
                                      single_pass_traversal_tag,
                                      ::cuda::std::_If<::cuda::std::is_convertible_v<Category, output_host_iterator_tag>,
                                                       incrementable_traversal_tag,
                                                       void>>>>>;

template <typename Category>
using device_system_category_to_traversal = ::cuda::std::_If<
  ::cuda::std::is_convertible_v<Category, random_access_device_iterator_tag>,
  random_access_traversal_tag,
  ::cuda::std::_If<
    ::cuda::std::is_convertible_v<Category, bidirectional_device_iterator_tag>,
    bidirectional_traversal_tag,
    ::cuda::std::_If<::cuda::std::is_convertible_v<Category, forward_device_iterator_tag>,
                     forward_traversal_tag,
                     ::cuda::std::_If<::cuda::std::is_convertible_v<Category, input_device_iterator_tag>,
                                      single_pass_traversal_tag,
                                      ::cuda::std::_If<::cuda::std::is_convertible_v<Category, output_device_iterator_tag>,
                                                       incrementable_traversal_tag,
                                                       void>>>>>;

template <typename Category>
using category_to_traversal =
  // check for host system
  ::cuda::std::_If<::cuda::std::is_convertible_v<Category, input_host_iterator_tag>
                     || ::cuda::std::is_convertible_v<Category, output_host_iterator_tag>,
                   host_system_category_to_traversal<Category>,
                   // check for device system
                   ::cuda::std::_If<::cuda::std::is_convertible_v<Category, input_device_iterator_tag>
                                      || ::cuda::std::is_convertible_v<Category, output_device_iterator_tag>,
                                    device_system_category_to_traversal<Category>,
                                    // unknown category
                                    void>>;

template <typename T>
_CCCL_INLINE_VAR constexpr bool is_iterator_traversal = ::cuda::std::is_convertible_v<T, incrementable_traversal_tag>;

template <typename CategoryOrTraversal>
struct iterator_category_to_traversal
{
  using type = ::cuda::std::
    _If<is_iterator_traversal<CategoryOrTraversal>, CategoryOrTraversal, category_to_traversal<CategoryOrTraversal>>;
};

} // namespace detail

THRUST_NAMESPACE_END

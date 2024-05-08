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
#include <thrust/iterator/iterator_categories.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <typename T>
struct is_host_iterator_category
    : thrust::detail::or_<::cuda::std::is_convertible<T, thrust::input_host_iterator_tag>,
                          ::cuda::std::is_convertible<T, thrust::output_host_iterator_tag>>
{}; // end is_host_iterator_category

template <typename T>
struct is_device_iterator_category
    : thrust::detail::or_<::cuda::std::is_convertible<T, thrust::input_device_iterator_tag>,
                          ::cuda::std::is_convertible<T, thrust::output_device_iterator_tag>>
{}; // end is_device_iterator_category

template <typename T>
struct is_iterator_category : thrust::detail::or_<is_host_iterator_category<T>, is_device_iterator_category<T>>
{}; // end is_iterator_category

} // namespace detail

THRUST_NAMESPACE_END

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

/*! \file generic/type_traits.h
 *  \brief Introspection for free functions defined in generic.
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

THRUST_NAMESPACE_BEGIN

// forward declaration of any_system_tag for any_conversion below
struct any_system_tag;

namespace system::detail::generic
{
template <typename, typename... Tags>
inline constexpr bool select_system_exists = false;

template <typename... Tags>
inline constexpr bool
  select_system_exists<::cuda::std::void_t<decltype(select_system(::cuda::std::declval<Tags>()...))>, Tags...> = true;

} // namespace system::detail::generic
THRUST_NAMESPACE_END

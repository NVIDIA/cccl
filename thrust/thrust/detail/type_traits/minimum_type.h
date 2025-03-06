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

THRUST_NAMESPACE_BEGIN

namespace detail
{

//
// Returns the minimum type or is empty
// if T1 and T2 are unrelated.
//
template <typename T1,
          typename T2,
          bool GreaterEqual = ::cuda::std::is_convertible_v<T1, T2>,
          bool LessEqual    = ::cuda::std::is_convertible_v<T2, T1>>
struct smaller_type
{};

// T1 >= T2
template <typename T1, typename T2>
struct smaller_type<T1, T2, true, false>
{
  using type = T2;
};

// T1 <= T2
template <typename T1, typename T2>
struct smaller_type<T1, T2, false, true>
{
  using type = T1;
};

template <typename T1, typename T2>
struct smaller_type<T1, T2, true, true>
{
  using type = T1;
};

template <typename T>
struct smaller_type<T, T, true, true>
{
  using type = T;
};

template <typename T>
struct smaller_type<T, T, false, false>
{
  using type = T;
};

template <typename Head, typename... Tail>
struct minimum_type_impl : smaller_type<Head, typename minimum_type_impl<Tail...>::type>
{};

template <typename T>
struct minimum_type_impl<T>
{
  using type = T;
};

// Alias to the minimum type of the given types. All types are convertible to the minimum type. If no such type exists,
// a SFINAE-friendly compile-time error is generated.
template <typename... Ts>
using minimum_type = typename minimum_type_impl<Ts...>::type;
} // namespace detail

THRUST_NAMESPACE_END

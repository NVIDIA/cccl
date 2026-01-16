/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

#include <cuda/std/__type_traits/void_t.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace pointer_traits_detail
{
template <typename Ptr, typename Enable = void>
struct pointer_raw_pointer_impl
{};

template <typename T>
struct pointer_raw_pointer_impl<T*>
{
  using type = T*;
};

template <typename Ptr>
struct pointer_raw_pointer_impl<Ptr, ::cuda::std::void_t<typename Ptr::raw_pointer>>
{
  using type = typename Ptr::raw_pointer;
};
} // namespace pointer_traits_detail

template <typename T>
struct pointer_raw_pointer : pointer_traits_detail::pointer_raw_pointer_impl<T>
{};

// Check whether we are dealing with either a raw pointer or a thrust smart pointer
template <typename T, typename = void>
inline constexpr bool is_thrust_pointer_v = false;

template <typename T>
inline constexpr bool is_thrust_pointer_v<T*> = true;

template <typename T>
inline constexpr bool is_thrust_pointer_v<T, ::cuda::std::void_t<typename T::raw_pointer>> = true;
} // namespace detail

THRUST_NAMESPACE_END

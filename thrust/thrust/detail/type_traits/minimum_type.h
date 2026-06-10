// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
struct no_minimum_type_marker
{};

// Returns the minimum type, or `no_minimum_type_marker`if T1 and T2 are unrelated.
template <typename T1,
          typename T2,
          bool GreaterEqual = ::cuda::std::is_convertible_v<T1, T2>,
          bool LessEqual    = ::cuda::std::is_convertible_v<T2, T1>>
struct smaller_type
{
  using type = T1;
};

// T1 >= T2
template <typename T1, typename T2>
struct smaller_type<T1, T2, true, false>
{
  using type = T2;
};

// unordered
template <typename T1, typename T2>
struct smaller_type<T1, T2, false, false>
{
  using type = no_minimum_type_marker;
};

template <typename Head, typename... Tail>
struct minimum_type_impl : smaller_type<Head, typename minimum_type_impl<Tail...>::type>
{};

template <typename T>
struct minimum_type_impl<T>
{
  using type = T;
};

// Has no nested ::type to produce a SFINAE-friendly error, in case the minimum_type is `no_minimum_type_marker`
template <typename SFINAE, typename... Ts>
struct minimum_type_check_marker
{};

template <typename... Ts>
struct minimum_type_check_marker<
  ::cuda::std::enable_if_t<!::cuda::std::is_same_v<typename minimum_type_impl<Ts...>::type, no_minimum_type_marker>>,
  Ts...> : minimum_type_impl<Ts...>
{};

// Alias to the minimum type of the given pack. The minimum type is the one to which all other types are convertible.
// If no such type exists, a SFINAE-friendly compile-time error is generated.
template <typename... Ts>
using minimum_type = typename minimum_type_check_marker<void, Ts...>::type;
} // namespace detail

THRUST_NAMESPACE_END

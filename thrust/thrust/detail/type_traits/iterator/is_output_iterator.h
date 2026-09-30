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

#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/void_t.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename SFINAE = void>
inline constexpr bool is_output_iterator = true;

template <typename T>
inline constexpr bool is_output_iterator<T, ::cuda::std::void_t<it_value_t<T>>> =
  ::cuda::std::is_void_v<it_value_t<T>> || ::cuda::std::is_same_v<it_value_t<T>, any_assign>;
} // namespace detail

THRUST_NAMESPACE_END

// SPDX-FileCopyrightText: Copyright (c) 2008-2022, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
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

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>

THRUST_NAMESPACE_BEGIN

// forward declaration of device_reference
template <typename T>
class device_reference;

namespace detail
{
/// helper classes [4.3].

template <typename T>
inline constexpr bool is_proxy_reference_v = false;

template <bool Condition, typename T = void>
using disable_if = ::cuda::std::enable_if<!Condition, T>;

template <typename T1, typename T2, typename T = void>
using enable_if_convertible_t = ::cuda::std::enable_if_t<::cuda::std::is_convertible_v<T1, T2>, T>;

template <typename T1, typename T2, typename T = void>
using disable_if_convertible = disable_if<::cuda::std::is_convertible_v<T1, T2>, T>;
} // namespace detail

using ::cuda::std::false_type;
using ::cuda::std::integral_constant;
using ::cuda::std::true_type;

THRUST_NAMESPACE_END

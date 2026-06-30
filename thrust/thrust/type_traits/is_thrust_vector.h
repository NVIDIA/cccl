// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file thrust/type_traits/is_thrust_vector.h
 *  \brief A type trait that determines if a type is a Thrust vector.
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is a
 *  Thrust vector and \c false otherwise.
 */
template <typename T>
inline constexpr bool is_thrust_vector_v = false;

template <typename T, typename Alloc>
inline constexpr bool is_thrust_vector_v<thrust::host_vector<T, Alloc>> = true;

template <typename T, typename Alloc>
inline constexpr bool is_thrust_vector_v<thrust::device_vector<T, Alloc>> = true;

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

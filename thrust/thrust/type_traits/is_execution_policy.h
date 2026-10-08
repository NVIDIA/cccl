// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file
 *  \brief A type trait that determines if a type is an \a ExecutionPolicy.
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

#include <thrust/detail/execution_policy.h>

#include <cuda/std/__type_traits/is_base_of.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c T is an \a ExecutionPolicy and \c false_type
 *  otherwise.
 */
template <typename T>
using is_execution_policy = ::cuda::std::is_base_of<detail::execution_policy_marker, T>;

/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is an
 *  \a ExecutionPolicy and \c false otherwise.
 */
template <typename T>
inline constexpr bool is_execution_policy_v = ::cuda::std::is_base_of_v<detail::execution_policy_marker, T>;

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

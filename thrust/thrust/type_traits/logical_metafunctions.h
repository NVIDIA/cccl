/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/negation.h>

THRUST_NAMESPACE_BEGIN

//! \addtogroup utility
//! \{
//! \addtogroup type_traits Type Traits
//! \{

using ::cuda::std::conjunction;
using ::cuda::std::disjunction;
using ::cuda::std::negation;
#if _CCCL_STD_VER >= 2014 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
using ::cuda::std::conjunction_v;
using ::cuda::std::disjunction_v;
using ::cuda::std::negation_v;
#endif

//! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
//! whose value is <tt>(... && Bs)</tt>.
//!
//! \see conjunction_value_v
//! \see conjunction
//! \see <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>
template <bool... Bs>
using conjunction_value = conjunction<::cuda::std::bool_constant<Bs>...>;

#if _CCCL_STD_VER >= 2014
//! \brief <tt>constexpr bool</tt> whose value is <tt>(... && Bs)</tt>.
//!
//! \see conjunction_value
//! \see conjunction
//! \see <a href="https://en.cppreference.com/w/cpp/types/conjunction"><tt>std::conjunction</tt></a>
template <bool... Bs>
constexpr bool conjunction_value_v = conjunction_value<Bs...>::value;
#endif

//! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
//! whose value is <tt>(... || Bs)</tt>.
//!
//! \see disjunction_value_v
//! \see disjunction
//! \see <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>
template <bool... Bs>
using disjunction_value = disjunction<::cuda::std::bool_constant<Bs>...>;

#if _CCCL_STD_VER >= 2014
//! \brief <tt>constexpr bool</tt> whose value is <tt>(... || Bs)</tt>.
//!
//! \see disjunction_value
//! \see disjunction
//! \see <a href="https://en.cppreference.com/w/cpp/types/disjunction"><tt>std::disjunction</tt></a>
template <bool... Bs>
constexpr bool disjunction_value_v = disjunction_value<Bs...>::value;
#endif

//! \brief <a href="https://en.cppreference.com/w/cpp/types/integral_constant"><tt>std::integral_constant</tt></a>
//! whose value is <tt>!Bs</tt>.
//!
//! \see negation_value_v
//! \see negation
//! \see <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
template <bool B>
using negation_value = ::cuda::std::bool_constant<!B>;

#if _CCCL_STD_VER >= 2014
//! \brief <tt>constexpr bool</tt> whose value is <tt>!Ts::value</tt>.
//!
//! \see negation_value
//! \see negation
//! \see <a href="https://en.cppreference.com/w/cpp/types/negation"><tt>std::negation</tt></a>
template <bool B>
constexpr bool negation_value_v = negation_value<B>::value;
#endif

//! \} // type traits
//! \} // utility

THRUST_NAMESPACE_END

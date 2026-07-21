// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file
 *  \brief <a href="https://wg21.link/P1144">P1144</a>'s proposed
 *  \c std::is_trivially_relocatable, an extensible type trait indicating
 *  whether a type can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>.
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
#include <thrust/detail/static_assert.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/is_same.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \cond
 */

namespace detail
{
template <typename T>
struct is_trivially_relocatable_impl;
} // namespace detail

/*! \endcond
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c T is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  aka can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  and \c false_type otherwise.
 *
 * \see is_trivially_relocatable_v
 * \see is_trivially_relocatable_to
 * \see is_indirectly_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename T>
using is_trivially_relocatable
  CCCL_DEPRECATED_BECAUSE("Use cuda::is_trivially_copyable instead") = detail::is_trivially_relocatable_impl<T>;

/*! \brief <tt>constexpr bool</tt> that is \c true if \c T is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  aka can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  and \c false otherwise.
 *
 * \see is_trivially_relocatable
 * \see is_trivially_relocatable_to
 * \see is_indirectly_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename T>
CCCL_DEPRECATED_BECAUSE("Use cuda::is_trivially_copyable_v instead") constexpr bool is_trivially_relocatable_v =
  detail::is_trivially_relocatable_impl<T>::value;

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/BinaryTypeTrait"><i>BinaryTypeTrait</i></a>
 *  that returns \c true_type if \c From is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  to \c To, aka can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  and \c false_type otherwise.
 *
 * \see is_trivially_relocatable_to_v
 * \see is_trivially_relocatable
 * \see is_indirectly_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename From, typename To>
using is_trivially_relocatable_to
  CCCL_DEPRECATED_BECAUSE("Use cuda::is_trivially_copyable and cuda::std::is_same instead") =
    integral_constant<bool, ::cuda::std::is_same_v<From, To> && detail::is_trivially_relocatable_impl<To>::value>;

/*! \brief <tt>constexpr bool</tt> that is \c true if \c From is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  to \c To, aka can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  and \c false otherwise.
 *
 * \see is_trivially_relocatable_to
 * \see is_trivially_relocatable
 * \see is_indirectly_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename From, typename To>
CCCL_DEPRECATED_BECAUSE("Use cuda::is_trivially_copyable_v and cuda::std::is_same_v instead") constexpr bool
  is_trivially_relocatable_to_v = ::cuda::std::is_same_v<From, To> && detail::is_trivially_relocatable_impl<To>::value;

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/BinaryTypeTrait"><i>BinaryTypeTrait</i></a>
 *  that returns \c true_type if the element type of \c FromIterator is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  to the element type of \c ToIterator, aka can be bitwise copied with a
 *  facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  and \c false_type otherwise.
 *
 * \see is_indirectly_trivially_relocatable_to_v
 * \see is_trivially_relocatable
 * \see is_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename FromIterator, typename ToIterator>
using is_indirectly_trivially_relocatable_to
  CCCL_DEPRECATED_BECAUSE("Use cuda::is_trivially_copyable, "
                          "cuda::std::is_same, and "
                          "thrust::is_contiguous_iterator instead") =
    integral_constant<bool,
                      is_contiguous_iterator_v<FromIterator>
                        && is_contiguous_iterator_v<ToIterator>&& ::cuda::std::
                          is_same_v<detail::it_value_t<FromIterator>, detail::it_value_t<ToIterator>>
                        && detail::is_trivially_relocatable_impl<detail::it_value_t<ToIterator>>::value>;

/*! \brief <tt>constexpr bool</tt> that is \c true if the element type of
 *  \c FromIterator is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  to the element type of \c ToIterator, aka can be bitwise copied with a
 *  facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  and \c false otherwise.
 *
 * \see is_indirectly_trivially_relocatable_to
 * \see is_trivially_relocatable
 * \see is_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename FromIterator, typename ToIterator>
CCCL_DEPRECATED_BECAUSE("Use cuda::is_trivially_copyable_v, cuda::std::is_same_v, and thrust::is_contiguous_iterator_v "
                        "instead") constexpr bool is_indirectly_trivially_relocate_to_v =
  is_contiguous_iterator_v<FromIterator> && is_contiguous_iterator_v<ToIterator>
  && ::cuda::std::is_same_v<detail::it_value_t<FromIterator>, detail::it_value_t<ToIterator>>
  && detail::is_trivially_relocatable_impl<detail::it_value_t<ToIterator>>::value;

namespace detail
{
template <typename FromIterator, typename ToIterator>
constexpr bool is_indirectly_trivially_copyable_to_v =
  is_contiguous_iterator_v<FromIterator> && is_contiguous_iterator_v<ToIterator>
  && ::cuda::std::is_same_v<it_value_t<FromIterator>, it_value_t<ToIterator>>
  && ::cuda::is_trivially_copyable_v<it_value_t<ToIterator>>;
}

/*! \brief <a href="http://eel.is/c++draft/namespace.std#def:customization_point"><i>customization point</i></a>
 *  that can be specialized customized to indicate that a type \c T is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  aka it can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>.
 *
 * \see is_indirectly_trivially_relocatable_to
 * \see is_trivially_relocatable
 * \see is_trivially_relocatable_to
 * \see THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE
 */
template <typename T>
struct CCCL_DEPRECATED_BECAUSE("Please specialize cuda::is_trivially_copyable_v instead")
proclaim_trivially_relocatable : false_type
{};

/*! \brief Declares that the type \c T is
 *  <a href="https://wg21.link/P1144"><i>TriviallyRelocatable</i></a>,
 *  aka it can be bitwise copied with a facility like
 *  <a href="https://en.cppreference.com/w/cpp/string/byte/memcpy"><tt>std::memcpy</tt></a>,
 *  by specializing \c proclaim_trivially_relocatable and \c cuda::is_trivially_copyable[_v].
 *
 * \see is_indirectly_trivially_relocatable_to
 * \see is_trivially_relocatable
 * \see is_trivially_relocatable_to
 * \see proclaim_trivially_relocatable
 * \see cuda::is_trivially_copyable_v
 */
#define THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(T)                            \
  THRUST_NAMESPACE_BEGIN                                                    \
  template <>                                                               \
  struct proclaim_trivially_relocatable<T> : THRUST_NS_QUALIFIER::true_type \
  {};                                                                       \
  THRUST_NAMESPACE_END                                                      \
  _CCCL_BEGIN_NAMESPACE_CUDA                                                \
  template <>                                                               \
  inline constexpr bool is_trivially_copyable_v<T> = true;                  \
  _CCCL_END_NAMESPACE_CUDA                                                  \
  /**/

///////////////////////////////////////////////////////////////////////////////

/*! \cond
 */

namespace detail
{
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
// https://wg21.link/P1144R0#wording-inheritance
template <typename T>
struct is_trivially_relocatable_impl
    : integral_constant<bool, ::cuda::is_trivially_copyable_v<T> || proclaim_trivially_relocatable<T>::value>
{};
_CCCL_SUPPRESS_DEPRECATED_POP

template <typename T, ::cuda::std::size_t N>
struct is_trivially_relocatable_impl<T[N]> : is_trivially_relocatable_impl<T>
{};
} // namespace detail

THRUST_NAMESPACE_END

// Note: the built-in CUDA vector types (e.g. float2), __half, and __half2 are already reported as
// trivially copyable by cuda::is_trivially_copyable, so is_trivially_relocatable_impl already treats
// them as trivially relocatable.

THRUST_NAMESPACE_BEGIN
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
template <typename T, typename U>
struct proclaim_trivially_relocatable<::cuda::std::pair<T, U>>
    : ::cuda::std::conjunction<detail::is_trivially_relocatable_impl<T>, detail::is_trivially_relocatable_impl<U>>
{};

template <typename... Ts>
struct proclaim_trivially_relocatable<::cuda::std::tuple<Ts...>>
    : ::cuda::std::conjunction<detail::is_trivially_relocatable_impl<Ts>...>
{};
_CCCL_SUPPRESS_DEPRECATED_POP
THRUST_NAMESPACE_END

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \} // type traits
 */

/*! \} // utility
 */

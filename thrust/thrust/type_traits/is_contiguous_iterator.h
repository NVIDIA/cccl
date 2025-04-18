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

/*! \file
 *  \brief An extensible type trait for determining if an iterator satisfies the
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *  requirements (aka is pointer-like).
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
#include <thrust/detail/type_traits/is_thrust_pointer.h>

#include <cuda/std/iterator>

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

template <typename Iterator>
struct is_contiguous_iterator_impl;

} // namespace detail

/*! \endcond
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that returns \c true_type if \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory, and \c false_type
 *  otherwise.
 *
 * \see is_contiguous_iterator_v
 * \see proclaim_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
using is_contiguous_iterator = detail::is_contiguous_iterator_impl<Iterator>;

/*! \brief <tt>constexpr bool</tt> that is \c true if \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory, and \c false
 *  otherwise.
 *
 * \see is_contiguous_iterator
 * \see proclaim_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<Iterator>::value;

/*! \brief Customization point that can be customized to indicate that an
 *  iterator type \c Iterator satisfies
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>,
 *  aka it points to elements that are contiguous in memory.
 *
 * \see is_contiguous_iterator
 * \see THRUST_PROCLAIM_CONTIGUOUS_ITERATOR
 */
template <typename Iterator>
struct proclaim_contiguous_iterator : false_type
{};

/*! \brief Declares that the iterator \c Iterator is
 *  <a href="https://en.cppreference.com/w/cpp/named_req/ContiguousIterator">ContiguousIterator</a>
 *  by specializing \c proclaim_contiguous_iterator.
 *
 * \see is_contiguous_iterator
 * \see proclaim_contiguous_iterator
 */
#define THRUST_PROCLAIM_CONTIGUOUS_ITERATOR(Iterator)                            \
  THRUST_NAMESPACE_BEGIN                                                         \
  template <>                                                                    \
  struct proclaim_contiguous_iterator<Iterator> : THRUST_NS_QUALIFIER::true_type \
  {};                                                                            \
  THRUST_NAMESPACE_END                                                           \
  /**/

/*! \cond
 */

namespace detail
{

template <typename Iterator>
struct is_libcxx_wrap_iter : false_type
{};

#if defined(_LIBCPP_VERSION)
template <typename Iterator>
struct is_libcxx_wrap_iter<
#  if _LIBCPP_VERSION < 14000
  _VSTD::__wrap_iter<Iterator>
#  else
  std::__wrap_iter<Iterator>
#  endif
  > : true_type
{};
#endif

template <typename Iterator>
struct is_libstdcxx_normal_iterator : false_type
{};

#if defined(__GLIBCXX__)
template <typename Iterator, typename Container>
struct is_libstdcxx_normal_iterator<::__gnu_cxx::__normal_iterator<Iterator, Container>> : true_type
{};
#endif

#if _CCCL_COMPILER(MSVC)

template <typename Iterator>
struct is_msvc_contiguous_iterator : ::cuda::std::is_pointer<::std::_Unwrapped_t<Iterator>>
{};

#else

template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type
{};

#endif

template <typename Iterator>
struct is_contiguous_iterator_impl
    : integral_constant<
        bool,
        ::cuda::std::contiguous_iterator<Iterator> || is_thrust_pointer<Iterator>::value
          || is_libcxx_wrap_iter<Iterator>::value || is_libstdcxx_normal_iterator<Iterator>::value
          || is_msvc_contiguous_iterator<Iterator>::value || proclaim_contiguous_iterator<Iterator>::value>
{};

} // namespace detail

/*! \endcond
 */

///////////////////////////////////////////////////////////////////////////////

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

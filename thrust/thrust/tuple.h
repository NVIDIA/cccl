/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements.
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <tuple>

THRUST_NAMESPACE_BEGIN

// define null_type for backwards compatability
struct null_type
{};

_CCCL_HOST_DEVICE inline bool operator==(const null_type&, const null_type&)
{
  return true;
}

_CCCL_HOST_DEVICE inline bool operator>=(const null_type&, const null_type&)
{
  return true;
}

_CCCL_HOST_DEVICE inline bool operator<=(const null_type&, const null_type&)
{
  return true;
}

_CCCL_HOST_DEVICE inline bool operator!=(const null_type&, const null_type&)
{
  return false;
}

_CCCL_HOST_DEVICE inline bool operator<(const null_type&, const null_type&)
{
  return false;
}

_CCCL_HOST_DEVICE inline bool operator>(const null_type&, const null_type&)
{
  return false;
}

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <size_t N, class T>
using tuple_element = _CUDA_VSTD::tuple_element<N, T>;

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <class T>
using tuple_size = _CUDA_VSTD::tuple_size<T>;

template <class>
struct __is_tuple_of_iterator_references : _CUDA_VSTD::false_type
{};

/*! \brief \p tuple is a class template that can be instantiated with up to ten
 *  arguments. Each template argument specifies the type of element in the \p
 *  tuple. Consequently, tuples are heterogeneous, fixed-size collections of
 *  values. An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class... Ts>
struct tuple : public _CUDA_VSTD::tuple<Ts...>
{
  using super_t = _CUDA_VSTD::tuple<Ts...>;
  using super_t::super_t;

  tuple() = default;

  template <class _TupleOfIteratorReferences,
            _CUDA_VSTD::__enable_if_t<__is_tuple_of_iterator_references<_TupleOfIteratorReferences>::value, int> = 0,
            _CUDA_VSTD::__enable_if_t<(tuple_size<_TupleOfIteratorReferences>::value == sizeof...(Ts)), int>     = 0>
  _CCCL_HOST_DEVICE tuple(_TupleOfIteratorReferences&& tup)
      : tuple(_CUDA_VSTD::forward<_TupleOfIteratorReferences>(tup).template __to_tuple<Ts...>(
          _CUDA_VSTD::__make_tuple_indices_t<sizeof...(Ts)>()))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class TupleLike,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__tuple_assignable<TupleLike, super_t>::value, int> = 0>
  _CCCL_HOST_DEVICE tuple& operator=(TupleLike&& other)
  {
    super_t::operator=(_CUDA_VSTD::forward<TupleLike>(other));
    return *this;
  }

#if defined(_CCCL_COMPILER_MSVC_2017)
  // MSVC2017 needs some help to convert tuples
  template <class... Us,
            _CUDA_VSTD::__enable_if_t<!_CUDA_VSTD::is_same<tuple<Us...>, tuple>::value, int> = 0,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__tuple_convertible<_CUDA_VSTD::tuple<Us...>, super_t>::value, int> = 0>
  _CCCL_HOST_DEVICE constexpr operator tuple<Us...>()
  {
    return __to_tuple<Us...>(typename _CUDA_VSTD::__make_tuple_indices<sizeof...(Ts)>::type{});
  }

  template <class... Us, size_t... Id>
  _CCCL_HOST_DEVICE constexpr tuple<Us...> __to_tuple(_CUDA_VSTD::__tuple_indices<Id...>) const
  {
    return tuple<Us...>{_CUDA_VSTD::get<Id>(*this)...};
  }
#endif // _CCCL_COMPILER_MSVC_2017
};

#if _CCCL_STD_VER >= 2017
template <class... Ts>
_CCCL_HOST_DEVICE tuple(Ts...) -> tuple<Ts...>;

template <class T1, class T2>
struct pair;

template <class T1, class T2>
_CCCL_HOST_DEVICE tuple(pair<T1, T2>) -> tuple<T1, T2>;
#endif // _CCCL_STD_VER >= 2017

template <class... Ts>
inline
  _CCCL_HOST_DEVICE _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__all<_CUDA_VSTD::__is_swappable<Ts>::value...>::value, void>
  swap(tuple<Ts...>& __x,
       tuple<Ts...>& __y) noexcept((_CUDA_VSTD::__all<_CUDA_VSTD::__is_nothrow_swappable<Ts>::value...>::value))
{
  __x.swap(__y);
}

template <class... Ts>
inline _CCCL_HOST_DEVICE tuple<typename _CUDA_VSTD::__unwrap_ref_decay<Ts>::type...> make_tuple(Ts&&... __t)
{
  return tuple<typename _CUDA_VSTD::__unwrap_ref_decay<Ts>::type...>(_CUDA_VSTD::forward<Ts>(__t)...);
}

template <class... Ts>
inline _CCCL_HOST_DEVICE tuple<Ts&...> tie(Ts&... ts) noexcept
{
  return tuple<Ts&...>(ts...);
}

using _CUDA_VSTD::get;

template <typename... Ts>
struct proclaim_trivially_relocatable<tuple<Ts...>> : ::cuda::std::conjunction<is_trivially_relocatable<Ts>...>
{};

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_size<tuple<Ts...>>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_element<Id, tuple<Ts...>>
{};

template <class... Ts>
struct __tuple_like_ext<THRUST_NS_QUALIFIER::tuple<Ts...>> : true_type
{};

template <>
struct tuple_size<tuple<THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<>>
{};

template <class T0>
struct tuple_size<tuple<T0,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0>>
{};

template <class T0, class T1>
struct tuple_size<tuple<T0,
                        T1,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1>>
{};

template <class T0, class T1, class T2>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2>>
{};

template <class T0, class T1, class T2, class T3>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        T3,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3>>
{};

template <class T0, class T1, class T2, class T3, class T4>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        T3,
                        T4,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        T3,
                        T4,
                        T5,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4, T5>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
struct tuple_size<
  tuple<T0, T1, T2, T3, T4, T5, T6, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>>
    : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>>
    : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, THRUST_NS_QUALIFIER::null_type>>
    : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>>
{};

_LIBCUDACXX_END_NAMESPACE_STD

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion
#if _CCCL_STD_VER >= 2017
namespace std
{
template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_size<tuple<Ts...>>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_element<Id, tuple<Ts...>>
{};
} // namespace std
#endif // _CCCL_STD_VER >= 2017

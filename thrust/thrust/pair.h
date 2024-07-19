/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file pair.h
 *  \brief A type encapsulating a heterogeneous pair of elements
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

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns either the type of a \p pair's
 *  \c first_type or \c second_type in its nested type, \c type.
 *
 *  \tparam N This parameter selects the member of interest.
 *  \tparam T A \c pair type of interest.
 */
template <size_t N, class T>
using tuple_element = _CUDA_VSTD::tuple_element<N, T>;

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns \c 2, the number of elements of a \p pair,
 *  in its nested data member, \c value.
 *
 *  \tparam Pair A \c pair type of interest.
 */
template <class T>
using tuple_size = _CUDA_VSTD::tuple_size<T>;

/*! \p pair is a generic data structure encapsulating a heterogeneous
 *  pair of values.
 *
 *  \tparam T1 The type of \p pair's first object type.  There are no
 *          requirements on the type of \p T1. <tt>T1</tt>'s type is
 *          provided by <tt>pair::first_type</tt>.
 *
 *  \tparam T2 The type of \p pair's second object type.  There are no
 *          requirements on the type of \p T2. <tt>T2</tt>'s type is
 *          provided by <tt>pair::second_type</tt>.
 */
template <class T, class U>
struct pair : public _CUDA_VSTD::pair<T, U>
{
  using super_t = _CUDA_VSTD::pair<T, U>;
  using super_t::super_t;

#if (defined(_CCCL_COMPILER_GCC) && __GNUC__ < 9) || (defined(_CCCL_COMPILER_CLANG) && __clang_major__ < 12)
  // For whatever reason nvcc complains about that constructor being used before being defined in a constexpr variable
  constexpr pair() = default;

  template <class _U1          = T,
            class _U2          = U,
            class _Constraints = typename _CUDA_VSTD::__pair_constraints<T, U>::template __constructible<_U1, _U2>,
            _CUDA_VSTD::__enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_HOST_DEVICE constexpr pair(_U1&& __u1, _U2&& __u2)
      : super_t(_CUDA_VSTD::forward<_U1>(__u1), _CUDA_VSTD::forward<_U2>(__u2))
  {}
#endif // _CCCL_COMPILER_GCC < 9 || _CCCL_COMPILER_CLANG < 12
};

#if _CCCL_STD_VER >= 2017
template <class _T1, class _T2>
_CCCL_HOST_DEVICE pair(_T1, _T2) -> pair<_T1, _T2>;
#endif // _CCCL_STD_VER >= 2017

template <class T1, class T2>
inline _CCCL_HOST_DEVICE
_CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__is_swappable<T1>::value && _CUDA_VSTD::__is_swappable<T2>::value, void>
swap(pair<T1, T2>& lhs, pair<T1, T2>& rhs) noexcept(
  (_CUDA_VSTD::__is_nothrow_swappable<T1>::value && _CUDA_VSTD::__is_nothrow_swappable<T2>::value))
{
  lhs.swap(rhs);
}

template <class T1, class T2>
inline _CCCL_HOST_DEVICE
pair<typename _CUDA_VSTD::__unwrap_ref_decay<T1>::type, typename _CUDA_VSTD::__unwrap_ref_decay<T2>::type>
make_pair(T1&& t1, T2&& t2)
{
  return pair<typename _CUDA_VSTD::__unwrap_ref_decay<T1>::type, typename _CUDA_VSTD::__unwrap_ref_decay<T2>::type>(
    _CUDA_VSTD::forward<T1>(t1), _CUDA_VSTD::forward<T2>(t2));
}

using _CUDA_VSTD::get;

template <typename T, typename U>
struct proclaim_trivially_relocatable<pair<T, U>>
    : ::cuda::std::conjunction<is_trivially_relocatable<T>, is_trivially_relocatable<U>>
{};

/*! \endcond
 */

/*! \} // pair
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class T1, class T2>
struct tuple_size<THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_size<pair<T1, T2>>
{};

template <size_t Id, class T1, class T2>
struct tuple_element<Id, THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_element<Id, pair<T1, T2>>
{};

template <class T1, class T2>
struct __tuple_like_ext<THRUST_NS_QUALIFIER::pair<T1, T2>> : true_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion
#if _CCCL_STD_VER >= 2017

#  include <utility>

namespace std
{
template <class T1, class T2>
struct tuple_size<THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_size<pair<T1, T2>>
{};

template <size_t Id, class T1, class T2>
struct tuple_element<Id, THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_element<Id, pair<T1, T2>>
{};
} // namespace std
#endif // _CCCL_STD_VER >= 2017

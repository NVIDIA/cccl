//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COMMON_TYPE_H
#define _LIBCUDACXX___TYPE_TRAITS_COMMON_TYPE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__type_traits/decay.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class... _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS common_type;

template <class ..._Tp>
using __common_type_t = typename common_type<_Tp...>::type;

// Let COND_RES(X, Y) be:
template <class _Tp, class _Up>
using __cond_type = decltype(false ? declval<_Tp>() : declval<_Up>());

#if _CCCL_STD_VER > 2017
template <class _Tp, class _Up, class = void>
struct __common_type3 {};

// sub-bullet 4 - "if COND_RES(CREF(D1), CREF(D2)) denotes a type..."
template <class _Tp, class _Up>
struct __common_type3<_Tp, _Up, void_t<__cond_type<const _Tp&, const _Up&>>>
{
    using type = remove_cvref_t<__cond_type<const _Tp&, const _Up&>>;
};

template <class _Tp, class _Up, class = void>
struct __common_type2_imp : __common_type3<_Tp, _Up> {};
#else
template <class _Tp, class _Up, class = void>
struct __common_type2_imp {};
#endif

// sub-bullet 3 - "if decay_t<decltype(false ? declval<D1>() : declval<D2>())> ..."
template <class _Tp, class _Up>
struct __common_type2_imp<_Tp, _Up, __void_t<__cond_type<_Tp, _Up>>>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE __decay_t<__cond_type<_Tp, _Up>> type;
};

template <class, class = void>
struct __common_type_impl {};

template <class... _Tp>
struct __common_types;

template <class _Tp, class _Up>
struct __common_type_impl<
    __common_types<_Tp, _Up>, __void_t<__common_type_t<_Tp, _Up>> >
{
  typedef __common_type_t<_Tp, _Up> type;
};

template <class _Tp, class _Up, class _Vp, class... _Rest>
struct __common_type_impl<__common_types<_Tp, _Up, _Vp, _Rest...>, __void_t<__common_type_t<_Tp, _Up>> >
    : __common_type_impl<__common_types<__common_type_t<_Tp, _Up>, _Vp, _Rest...>> {};

// bullet 1 - sizeof...(Tp) == 0

template <>
struct _LIBCUDACXX_TEMPLATE_VIS common_type<> {};

// bullet 2 - sizeof...(Tp) == 1

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS common_type<_Tp>
    : public common_type<_Tp, _Tp> {};

// bullet 3 - sizeof...(Tp) == 2

// sub-bullet 1 - "If is_same_v<T1, D1> is false or ..."
template <class _Tp, class _Up, class _D1 = __decay_t<_Tp>, class _D2 = __decay_t<_Up>>
struct __common_type2 : common_type<_D1, _D2> {};

template <class _Tp, class _Up>
struct __common_type2<_Tp, _Up, _Tp, _Up> : __common_type2_imp<_Tp, _Up> {};

template <class _Tp, class _Up>
struct _LIBCUDACXX_TEMPLATE_VIS common_type<_Tp, _Up>
    : __common_type2<_Tp, _Up> {};

// bullet 4 - sizeof...(Tp) > 2

template <class _Tp, class _Up, class _Vp, class... _Rest>
struct _LIBCUDACXX_TEMPLATE_VIS common_type<_Tp, _Up, _Vp, _Rest...>
    : __common_type_impl<__common_types<_Tp, _Up, _Vp, _Rest...> > {};

#if _CCCL_STD_VER > 2011
template <class ..._Tp> using common_type_t = typename common_type<_Tp...>::type;

template<class, class, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __has_common_type = false;

template<class _Tp, class _Up>
_LIBCUDACXX_INLINE_VAR constexpr bool __has_common_type<_Tp, _Up, void_t<common_type_t<_Tp, _Up>>> = true;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_COMMON_TYPE_H

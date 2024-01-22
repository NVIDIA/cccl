//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POLYMORPHIC_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POLYMORPHIC_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_class.h" // __two
#include "../__utility/declval.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_POLYMORPHIC) && !defined(_LIBCUDACXX_USE_IS_POLYMORPHIC_FALLBACK)

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_polymorphic
    : public integral_constant<bool, _LIBCUDACXX_IS_POLYMORPHIC(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_polymorphic_v = _LIBCUDACXX_IS_POLYMORPHIC(_Tp);
#endif

#else

template<typename _Tp>
_LIBCUDACXX_HOST_DEVICE char &__is_polymorphic_impl(__enable_if_t<sizeof((_Tp*)dynamic_cast<const volatile void*>(_CUDA_VSTD::declval<_Tp*>())) != 0, int>);
template<typename _Tp>
_LIBCUDACXX_HOST_DEVICE __two &__is_polymorphic_impl(...);

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_polymorphic
    : public integral_constant<bool, sizeof(__is_polymorphic_impl<_Tp>(0)) == 1> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_polymorphic_v = is_polymorphic<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_POLYMORPHIC) && !defined(_LIBCUDACXX_USE_IS_POLYMORPHIC_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POLYMORPHIC_H

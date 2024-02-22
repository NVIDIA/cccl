//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
#define _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_destructible.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_nothrow_destructible.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011

#if defined(_CCCL_COMPILER_MSVC)

template<class _Tp>
_LIBCUDACXX_CONCEPT destructible = __is_nothrow_destructible(_Tp);

#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv

template<class _Tp, class = void, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __destructible_impl = false;

template<class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __destructible_impl<_Tp,
                                   __enable_if_t<_LIBCUDACXX_TRAIT(is_object, _Tp)>,
#if defined(_CCCL_COMPILER_GCC)
                                   __enable_if_t<_LIBCUDACXX_TRAIT(is_destructible, _Tp)>>
#else // ^^^ _CCCL_COMPILER_GCC ^^^ / vvv !_CCCL_COMPILER_GCC vvv
                                   __void_t<decltype(_CUDA_VSTD::declval<_Tp>().~_Tp())>>
#endif // !_CCCL_COMPILER_GCC
                                   = noexcept(_CUDA_VSTD::declval<_Tp>().~_Tp());

template<class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __destructible = __destructible_impl<_Tp>;

template<class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __destructible<_Tp&> = true;

template<class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __destructible<_Tp&&> = true;

template<class _Tp, size_t _Nm>
_LIBCUDACXX_INLINE_VAR constexpr bool __destructible<_Tp[_Nm]> = __destructible<_Tp>;

template<class _Tp>
_LIBCUDACXX_CONCEPT destructible = __destructible<_Tp>;

#endif // !_CCCL_COMPILER_MSVC

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H

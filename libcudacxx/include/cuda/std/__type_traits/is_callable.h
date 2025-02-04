//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Func, class... _Args>
using __call_result_t = decltype(_CUDA_VSTD::declval<_Func>()(_CUDA_VSTD::declval<_Args>()...));

template <class _Func, class... _Args>
struct __is_callable : _IsValidExpansion<__call_result_t, _Func, _Args...>
{};

#ifndef _CCCL_NO_VARIABLE_TEMPLATES
template <class _Func, class... _Args>
_CCCL_INLINE_VAR constexpr bool __is_callable_v = _IsValidExpansion<__call_result_t, _Func, _Args...>::value;
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H

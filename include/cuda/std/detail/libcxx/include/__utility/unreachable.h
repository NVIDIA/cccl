//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_UNREACHABLE_H
#define _LIBCUDACXX___UTILITY_UNREACHABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#ifndef _LIBCUDACXX_UNREACHABLE
#ifdef __GNUC__
#  define _LIBCUDACXX_UNREACHABLE() __builtin_unreachable()
#elif __has_builtin(__builtin_unreachable)
#  define _LIBCUDACXX_UNREACHABLE() __builtin_unreachable()
#else
#ifdef __CUDA_ARCH__
#  define _LIBCUDACXX_UNREACHABLE() __trap()
#else
#  define _LIBCUDACXX_UNREACHABLE() _CUDA_VSTD::abort()
#endif
#endif // has_builtin
#endif // !_LIBCUDACXX_UNREACHABLE

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_NORETURN _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
inline void __libcpp_unreachable()
{
  _LIBCUDACXX_UNREACHABLE();
}

#if _LIBCUDACXX_STD_VER > 20

[[noreturn]] _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
inline void unreachable() { _LIBCUDACXX_UNREACHABLE(); }

#endif // _LIBCUDACXX_STD_VER > 20

_LIBCUDACXX_END_NAMESPACE_STD

#endif

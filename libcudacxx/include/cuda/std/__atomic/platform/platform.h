// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#if defined(_CCCL_COMPILER_MSVC)
#include <cuda/std/__atomic/platform/msvc_to_builtins.h>
#endif

#if defined(__CLANG_ATOMIC_BOOL_LOCK_FREE)
# define ATOMIC_BOOL_LOCK_FREE      __CLANG_ATOMIC_BOOL_LOCK_FREE
# define ATOMIC_CHAR_LOCK_FREE      __CLANG_ATOMIC_CHAR_LOCK_FREE
# define ATOMIC_CHAR16_T_LOCK_FREE  __CLANG_ATOMIC_CHAR16_T_LOCK_FREE
# define ATOMIC_CHAR32_T_LOCK_FREE  __CLANG_ATOMIC_CHAR32_T_LOCK_FREE
# define ATOMIC_WCHAR_T_LOCK_FREE   __CLANG_ATOMIC_WCHAR_T_LOCK_FREE
# define ATOMIC_SHORT_LOCK_FREE     __CLANG_ATOMIC_SHORT_LOCK_FREE
# define ATOMIC_INT_LOCK_FREE       __CLANG_ATOMIC_INT_LOCK_FREE
# define ATOMIC_LONG_LOCK_FREE      __CLANG_ATOMIC_LONG_LOCK_FREE
# define ATOMIC_LLONG_LOCK_FREE     __CLANG_ATOMIC_LLONG_LOCK_FREE
# define ATOMIC_POINTER_LOCK_FREE   __CLANG_ATOMIC_POINTER_LOCK_FREE
#elif defined(__GCC_ATOMIC_BOOL_LOCK_FREE)
# define ATOMIC_BOOL_LOCK_FREE      __GCC_ATOMIC_BOOL_LOCK_FREE
# define ATOMIC_CHAR_LOCK_FREE      __GCC_ATOMIC_CHAR_LOCK_FREE
# define ATOMIC_CHAR16_T_LOCK_FREE  __GCC_ATOMIC_CHAR16_T_LOCK_FREE
# define ATOMIC_CHAR32_T_LOCK_FREE  __GCC_ATOMIC_CHAR32_T_LOCK_FREE
# define ATOMIC_WCHAR_T_LOCK_FREE   __GCC_ATOMIC_WCHAR_T_LOCK_FREE
# define ATOMIC_SHORT_LOCK_FREE     __GCC_ATOMIC_SHORT_LOCK_FREE
# define ATOMIC_INT_LOCK_FREE       __GCC_ATOMIC_INT_LOCK_FREE
# define ATOMIC_LONG_LOCK_FREE      __GCC_ATOMIC_LONG_LOCK_FREE
# define ATOMIC_LLONG_LOCK_FREE     __GCC_ATOMIC_LLONG_LOCK_FREE
# define ATOMIC_POINTER_LOCK_FREE   __GCC_ATOMIC_POINTER_LOCK_FREE
#endif

#if !defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)
#define ATOMIC_BOOL_LOCK_FREE      2
#define ATOMIC_CHAR_LOCK_FREE      2
#define ATOMIC_CHAR16_T_LOCK_FREE  2
#define ATOMIC_CHAR32_T_LOCK_FREE  2
#define ATOMIC_WCHAR_T_LOCK_FREE   2
#define ATOMIC_SHORT_LOCK_FREE     2
#define ATOMIC_INT_LOCK_FREE       2
#define ATOMIC_LONG_LOCK_FREE      2
#define ATOMIC_LLONG_LOCK_FREE     2
#define ATOMIC_POINTER_LOCK_FREE   2
#endif //!defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)

#if defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)
template<typename _Tp> struct __atomic_is_always_lock_free {
    enum { __value = _LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE(sizeof(_Tp), 0) }; };
#else
template<typename _Tp> struct __atomic_is_always_lock_free {
    enum { __value = sizeof(_Tp) <= 8 }; };
#endif // defined(_LIBCUDACXX_ATOMIC_ALWAYS_LOCK_FREE)

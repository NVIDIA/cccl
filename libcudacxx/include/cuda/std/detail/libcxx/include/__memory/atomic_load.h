// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ATOMIMC_LOAD_H
#define _LIBCUDACXX___MEMORY_ATOMIMC_LOAD_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../atomic"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef __cuda_std__

template <class _ValueType>
inline _LIBCUDACXX_INLINE_VISIBILITY
_ValueType __libcpp_relaxed_load(_ValueType const* __value) {
#if !defined(_LIBCUDACXX_HAS_NO_THREADS) && \
    defined(__ATOMIC_RELAXED) &&        \
    (__has_builtin(__atomic_load_n) || defined(_LIBCUDACXX_COMPILER_GCC))
    return __atomic_load_n(__value, __ATOMIC_RELAXED);
#else
    return *__value;
#endif
}

template <class _ValueType>
inline _LIBCUDACXX_INLINE_VISIBILITY
_ValueType __libcpp_acquire_load(_ValueType const* __value) {
#if !defined(_LIBCUDACXX_HAS_NO_THREADS) && \
    defined(__ATOMIC_ACQUIRE) &&        \
    (__has_builtin(__atomic_load_n) || defined(_LIBCUDACXX_COMPILER_GCC))
    return __atomic_load_n(__value, __ATOMIC_ACQUIRE);
#else
    return *__value;
#endif
}

#else

template <class _ValueType>
inline _LIBCUDACXX_INLINE_VISIBILITY
_ValueType __libcpp_relaxed_load(atomic<_ValueType> const* __value) {
    return __value->load(memory_order_relaxed);
}

template <class _ValueType>
inline _LIBCUDACXX_INLINE_VISIBILITY
_ValueType __libcpp_acquire_load(atomic<_ValueType> const* __value) {
    return __value->load(memory_order_acquire);
}
#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_ATOMIMC_LOAD_H

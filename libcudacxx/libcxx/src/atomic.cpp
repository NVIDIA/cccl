//===------------------------- atomic.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include "atomic"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if !defined(_LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE)

__libcpp_contention_t __libcpp_contention_state_[ 256 /* < there's no magic in this number */ ];

_LIBCUDACXX_FUNC_VIS
__libcpp_contention_t * __libcpp_contention_state(void const volatile * p) noexcept {
    return __libcpp_contention_state_ + ((std::uintptr_t)p & 255);
}

#endif //_LIBCUDACXX_HAS_NO_THREAD_CONTENTION_TABLE

_LIBCUDACXX_END_NAMESPACE_STD

#endif //_LIBCUDACXX_HAS_NO_THREADS

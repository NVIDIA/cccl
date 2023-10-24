//===------------------------- barrier.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_HAS_NO_THREADS
#include "barrier"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if !defined(_LIBCUDACXX_HAS_NO_TREE_BARRIER) && !defined(_LIBCUDACXX_HAS_NO_THREAD_FAVORITE_BARRIER_INDEX)

_LIBCUDACXX_EXPORTED_FROM_ABI
thread_local ptrdiff_t __libcpp_thread_favorite_barrier_index = 0;

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif //_LIBCUDACXX_HAS_NO_THREADS

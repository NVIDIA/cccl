// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_ATOMIC_PRELUDE_H
#define _LIBCUDACXX___CUDA_ATOMIC_PRELUDE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _CCCL_COMPILER_NVRTC
#  include <cuda/std/cassert> // TRANSITION: Fix transitive includes

#  include <atomic>
static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "");
static_assert(ATOMIC_CHAR_LOCK_FREE == 2, "");
static_assert(ATOMIC_CHAR16_T_LOCK_FREE == 2, "");
static_assert(ATOMIC_CHAR32_T_LOCK_FREE == 2, "");
static_assert(ATOMIC_WCHAR_T_LOCK_FREE == 2, "");
static_assert(ATOMIC_SHORT_LOCK_FREE == 2, "");
static_assert(ATOMIC_INT_LOCK_FREE == 2, "");
static_assert(ATOMIC_LONG_LOCK_FREE == 2, "");
static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "");
static_assert(ATOMIC_POINTER_LOCK_FREE == 2, "");
#  undef ATOMIC_BOOL_LOCK_FREE
#  undef ATOMIC_BOOL_LOCK_FREE
#  undef ATOMIC_CHAR_LOCK_FREE
#  undef ATOMIC_CHAR16_T_LOCK_FREE
#  undef ATOMIC_CHAR32_T_LOCK_FREE
#  undef ATOMIC_WCHAR_T_LOCK_FREE
#  undef ATOMIC_SHORT_LOCK_FREE
#  undef ATOMIC_INT_LOCK_FREE
#  undef ATOMIC_LONG_LOCK_FREE
#  undef ATOMIC_LLONG_LOCK_FREE
#  undef ATOMIC_POINTER_LOCK_FREE
#  undef ATOMIC_FLAG_INIT
#  undef ATOMIC_VAR_INIT
#endif // _CCCL_COMPILER_NVRTC

// pre-define lock free query for heterogeneous compatibility
#ifndef _LIBCUDACXX_ATOMIC_IS_LOCK_FREE
#  define _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(__x) (__x <= 8)
#endif

#ifndef _CCCL_COMPILER_NVRTC
#  include <thread>

#  include <errno.h>
#endif // _CCCL_COMPILER_NVRTC

#endif // _LIBCUDACXX___CUDA_ATOMIC_PRELUDE_H

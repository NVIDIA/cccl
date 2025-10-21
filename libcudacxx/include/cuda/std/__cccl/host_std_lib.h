//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_HOST_STD_LIB_H
#define __CCCL_HOST_STD_LIB_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/preprocessor.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _CCCL_HOST_STD_LIB_LIBSTDCXX() 0
#define _CCCL_HOST_STD_LIB_LIBCXX()    0
#define _CCCL_HOST_STD_LIB_STL()       0

// include a minimal header
#if _CCCL_HAS_INCLUDE(<version>)
#  include <version>
#elif _CCCL_HAS_INCLUDE(<ciso646>)
#  include <ciso646>
#endif // ^^^ _CCCL_HAS_INCLUDE(<ciso646>) ^^^

#if defined(_MSVC_STL_VERSION)
#  undef _CCCL_HOST_STD_LIB_STL
#  define _CCCL_HOST_STD_LIB_STL() 1
#elif defined(__GLIBCXX__)
#  undef _CCCL_HOST_STD_LIB_LIBSTDCXX
#  define _CCCL_HOST_STD_LIB_LIBSTDCXX() 1
#elif defined(_LIBCPP_VERSION)
#  undef _CCCL_HOST_STD_LIB_LIBCXX
#  define _CCCL_HOST_STD_LIB_LIBCXX() 1
#endif // ^^^ _LIBCPP_VERSION ^^^

#define _CCCL_HOST_STD_LIB(_X) _CCCL_HOST_STD_LIB_##_X()
#define _CCCL_HAS_HOST_STD_LIB() \
  (_CCCL_HOST_STD_LIB_LIBSTDCXX() || _CCCL_HOST_STD_LIB_LIBCXX() || _CCCL_HOST_STD_LIB_STL())

#endif // __CCCL_HOST_STD_LIB_H

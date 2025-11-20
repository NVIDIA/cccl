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

#define _CCCL_HOST_STD_LIB_LIBSTDCXX() _CCCL_VERSION_INVALID()
#define _CCCL_HOST_STD_LIB_LIBCXX()    _CCCL_VERSION_INVALID()
#define _CCCL_HOST_STD_LIB_STL()       _CCCL_VERSION_INVALID()

// include a minimal header
#if _CCCL_HAS_INCLUDE(<version>)
#  include <version>
#elif _CCCL_HAS_INCLUDE(<ciso646>)
#  include <ciso646>
#endif // ^^^ _CCCL_HAS_INCLUDE(<ciso646>) ^^^

#define _CCCL_HOST_STD_LIB_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 100 + (_MINOR))
#define _CCCL_HOST_STD_LIB(...)                         _CCCL_VERSION_COMPARE(_CCCL_HOST_STD_LIB_, _CCCL_HOST_STD_LIB_##__VA_ARGS__)

#if defined(_MSVC_STL_VERSION)
#  undef _CCCL_HOST_STD_LIB_STL
#  define _CCCL_HOST_STD_LIB_STL() (_MSVC_STL_VERSION, 0)
#elif defined(__GLIBCXX__)
#  undef _CCCL_HOST_STD_LIB_LIBSTDCXX
#  define _CCCL_HOST_STD_LIB_LIBSTDCXX() (_GLIBCXX_RELEASE, 0)
#elif defined(_LIBCPP_VERSION)
#  undef _CCCL_HOST_STD_LIB_LIBCXX
// since llvm-16, the version scheme has been changed from MMppp to MMmmpp
#  if _LIBCPP_VERSION / 10000 < 2
#    define _CCCL_HOST_STD_LIB_LIBCXX() (_LIBCPP_VERSION / 1000, 0)
#  else
#    define _CCCL_HOST_STD_LIB_LIBCXX() (_LIBCPP_VERSION / 10000, (_LIBCPP_VERSION / 100) % 100)
#  endif
#endif // ^^^ _LIBCPP_VERSION ^^^

#define _CCCL_HAS_HOST_STD_LIB() \
  (_CCCL_HOST_STD_LIB(LIBSTDCXX) || _CCCL_HOST_STD_LIB(LIBCXX) || _CCCL_HOST_STD_LIB(STL))

#endif // __CCCL_HOST_STD_LIB_H

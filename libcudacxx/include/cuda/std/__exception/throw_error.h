//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXCEPTION_THROW_ERROR_H
#define _CUDA_STD___EXCEPTION_THROW_ERROR_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__exception/terminate.h>

#include <nv/target>

// libstdc++ and the MSVC STL throw these exceptions from their runtime library, which saves us from including
// <stdexcept>, because that includes all of <string>
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_STD_LIB(LIBSTDCXX) && __has_include(<bits/stdexcept_throw.h>)
#  include <bits/new_throw.h>
#  include <bits/stdexcept_throw.h>
#elif _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_STD_LIB(LIBSTDCXX) && __has_include(<bits/functexcept.h>)
#  include <bits/functexcept.h>
#elif _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_STD_LIB(STL)
#  include <xutility>
#elif _CCCL_HOST_STD_LIB(LIBCXX) && __has_include(<__new/exceptions.h>)
#  include <cuda/std/__host_stdlib/stdexcept> // libc++ has no smaller header for these exceptions

#  include <__new/exceptions.h>
#else // ^^^ libc++ 20 and newer ^^^ / vvv other host standard library or no exceptions vvv
#  include <cuda/std/__host_stdlib/new>
#  include <cuda/std/__host_stdlib/stdexcept>
#endif // ^^^ other host standard library or no exceptions ^^^

#include <cuda/std/__cccl/prologue.h>

// Behaves like _CCCL_THROW(::std::_TYPE, __VA_ARGS__)
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION() && (_CCCL_HOST_STD_LIB(LIBSTDCXX) || _CCCL_HOST_STD_LIB(STL))
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX)
#    define _CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, ...) ::std::__throw_##_TYPE(__VA_ARGS__)
#  else // ^^^ _CCCL_HOST_STD_LIB(LIBSTDCXX) ^^^ / vvv _CCCL_HOST_STD_LIB(STL) vvv
#    define _CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, ...) ::std::_X##_TYPE(__VA_ARGS__)
#  endif // ^^^ _CCCL_HOST_STD_LIB(STL) ^^^
#  define _CCCL_THROW_STD(_TYPE, ...)                                                                                   \
    do                                                                                                                  \
    {                                                                                                                   \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (_CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, __VA_ARGS__);), (::cuda::std::terminate();)) \
    } while (0)
#else // ^^^ throw through the host standard library ^^^ / vvv _CCCL_THROW vvv
#  define _CCCL_THROW_STD(_TYPE, ...) _CCCL_THROW(::std::_TYPE, __VA_ARGS__)
#endif // ^^^ _CCCL_THROW ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_THROW_ERROR_H

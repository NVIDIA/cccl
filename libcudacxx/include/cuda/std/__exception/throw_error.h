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

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__host_stdlib/cstdio> // IWYU pragma: keep

#include <nv/target>

// libstdc++ and the MSVC STL throw these exceptions from their runtime library, which saves us from including
// <stdexcept>, because that includes all of <string>
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOSTED()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX)
#    if __has_include(<bits/stdexcept_throw.h>)
#      include <bits/new_throw.h>
#      include <bits/stdexcept_throw.h>
#    else // ^^^ libstdc++ 16 and newer ^^^ / vvv libstdc++ 15 and older vvv
#      include <bits/functexcept.h>
#    endif // ^^^ libstdc++ 15 and older ^^^
#  elif _CCCL_HOST_STD_LIB(STL)
#    include <xutility>
#  else // ^^^ _CCCL_HOST_STD_LIB(STL) ^^^ / vvv other host standard library vvv
#    include <cuda/std/__host_stdlib/new>
#    include <cuda/std/__host_stdlib/stdexcept>
#  endif // ^^^ other host standard library ^^^
#endif // _CCCL_HAS_EXCEPTIONS() && _CCCL_HOSTED()

#include <cuda/std/__cccl/prologue.h>

// Behaves like _CCCL_THROW(::std::_TYPE, __VA_ARGS__), where _WHAT is the message of the exception
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOSTED() && _CCCL_HOST_COMPILATION()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX)
#    define _CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, ...) ::std::__throw_##_TYPE(__VA_ARGS__)
#  elif _CCCL_HOST_STD_LIB(STL)
#    define _CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, ...) ::std::_X##_TYPE(__VA_ARGS__)
#  else // ^^^ _CCCL_HOST_STD_LIB(STL) ^^^ / vvv other host standard library vvv
#    define _CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, ...) throw ::std::_TYPE(__VA_ARGS__)
#  endif // ^^^ other host standard library ^^^
#  define _CCCL_THROW_STD_EXCEPTION(_TYPE, _WHAT, ...)                                                                  \
    do                                                                                                                  \
    {                                                                                                                   \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (_CCCL_THROW_HOST_STD_EXCEPTION(_TYPE, __VA_ARGS__);), (::cuda::std::terminate();)) \
    } while (0)
#elif _CCCL_HOSTJIT()
#  define _CCCL_THROW_STD_EXCEPTION(_TYPE, _WHAT, ...)                              \
    do                                                                              \
    {                                                                               \
      _CCCL_ASSERT(false, "An instance of class std::" #_TYPE " would be thrown."); \
      ::cuda::std::terminate();                                                     \
    } while (0)
#else // ^^^ _CCCL_HOSTJIT() ^^^ / vvv no exceptions vvv
#  define _CCCL_THROW_STD_EXCEPTION(_TYPE, _WHAT, ...)                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
      NV_IF_ELSE_TARGET(                                                                                               \
        NV_IS_HOST,                                                                                                    \
        ({                                                                                                             \
          ::fprintf(stderr, "An instance of class std::" #_TYPE " would be thrown.\n  what():  %s\nAborted\n", _WHAT); \
          ::fflush(stderr);                                                                                            \
        }),                                                                                                            \
        ({ _CCCL_ASSERT(false, "An instance of class std::" #_TYPE " would be thrown."); }))                           \
      ::cuda::std::terminate();                                                                                        \
    } while (0)
#endif // ^^^ no exceptions ^^^

_CCCL_BEGIN_NAMESPACE_CUDA_STD

[[noreturn]] _CCCL_API inline void __throw_bad_alloc()
{
  _CCCL_THROW_STD_EXCEPTION(bad_alloc, "std::bad_alloc", );
}

[[noreturn]] _CCCL_API inline void __throw_invalid_argument([[maybe_unused]] const char* __msg)
{
  _CCCL_THROW_STD_EXCEPTION(invalid_argument, __msg, __msg);
}

[[noreturn]] _CCCL_API inline void __throw_length_error([[maybe_unused]] const char* __msg)
{
  _CCCL_THROW_STD_EXCEPTION(length_error, __msg, __msg);
}

[[noreturn]] _CCCL_API inline void __throw_out_of_range([[maybe_unused]] const char* __msg)
{
  _CCCL_THROW_STD_EXCEPTION(out_of_range, __msg, __msg);
}

[[noreturn]] _CCCL_API inline void __throw_overflow_error([[maybe_unused]] const char* __msg)
{
  _CCCL_THROW_STD_EXCEPTION(overflow_error, __msg, __msg);
}

[[noreturn]] _CCCL_API inline void __throw_runtime_error([[maybe_unused]] const char* __msg)
{
  _CCCL_THROW_STD_EXCEPTION(runtime_error, __msg, __msg);
}

_CCCL_END_NAMESPACE_CUDA_STD

#undef _CCCL_THROW_STD_EXCEPTION
#undef _CCCL_THROW_HOST_STD_EXCEPTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_THROW_ERROR_H

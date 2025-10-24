//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXCEPTION_EXCEPTION_MACROS_H
#define _CUDA_STD___EXCEPTION_EXCEPTION_MACROS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __cccl_catch_any_lvalue
{
  template <class _Tp>
  _CCCL_HOST_DEVICE operator _Tp&() const noexcept;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

// The following macros are used to conditionally compile exception handling code. They
// are used in the same way as `try` and `catch`, but they allow for different behavior
// based on whether exceptions are enabled or not, and whether the code is being compiled
// for device or not.
//
// Usage:
//   _CCCL_TRY
//   {
//     can_throw();               // Code that may throw an exception
//   }
//   _CCCL_CATCH (cuda_error& e)  // Handle CUDA exceptions
//   {
//     printf("CUDA error: %s\n", e.what());
//   }
//   _CCCL_CATCH_ALL              // Handle any other exceptions
//   {
//     printf("unknown error\n");
//   }
//
// Notes:
//   - try/catch blocks mustn't be nested, because of possible variable shadowing
//   - if using try/catch block inside an if statement, always wrap it in a scope
//   - the catch clause must always bind to a named variable

#if _CCCL_HAS_EXCEPTIONS()
#  define _CCCL_TRY       try
#  define _CCCL_CATCH     catch
#  define _CCCL_CATCH_ALL catch (...)
#  define _CCCL_CATCH_FALLTHROUGH
#  if _CCCL_CUDA_COMPILER(NVHPC)
#    define _CCCL_THROW(...)                                                             \
      do                                                                                 \
      {                                                                                  \
        NV_IF_ELSE_TARGET(NV_IS_HOST, (throw __VA_ARGS__;), (::cuda::std::terminate();)) \
      } while (false)
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
#    define _CCCL_THROW(...) throw __VA_ARGS__
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_RETHROW throw
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
#  define _CCCL_TRY                                                                                     \
    if constexpr (true) \
    {
#  define _CCCL_CATCH(...)                                                       \
    }                                                                            \
    else if constexpr (false) for (__VA_ARGS__ = ::cuda::std::__cccl_catch_any_lvalue{}; false;) \
    {
#  define _CCCL_CATCH_ALL \
    }                     \
    else
#  define _CCCL_CATCH_FALLTHROUGH \
    }                             \
    else                          \
    {                             \
    }
#  define _CCCL_THROW(...) ::cuda::std::terminate()
#  define _CCCL_RETHROW    ::cuda::std::terminate()
#endif // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^

#endif // _CUDA_STD___EXCEPTION_EXCEPTION_MACROS_H

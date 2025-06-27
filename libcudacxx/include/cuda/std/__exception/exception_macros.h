//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___EXCEPTION_EXCEPTION_MACROS_H
#define _LIBCUDACXX___EXCEPTION_EXCEPTION_MACROS_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __cccl_catch_any_lvalue
{
  template <class _Tp>
  _CCCL_HOST_DEVICE operator _Tp&() const noexcept;

  template <class _Tp>
  _CCCL_HOST_DEVICE operator _Tp&&() const noexcept = delete;
};

_LIBCUDACXX_END_NAMESPACE_STD

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
#if _CCCL_HAS_EXCEPTIONS()
#  define _CCCL_TRY       try
#  define _CCCL_CATCH     catch
#  define _CCCL_CATCH_ALL catch (...)
#  define _CCCL_THROW(...)                                                                      \
    do                                                                                          \
    {                                                                                           \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (throw __VA_ARGS__;), (_CUDA_VSTD_NOVERSION::terminate();)) \
    } while (false)
#  define _CCCL_RETHROW                                                             \
    do                                                                              \
    {                                                                               \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (throw;), (_CUDA_VSTD_NOVERSION::terminate();)) \
    } while (false)
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
#  define _CCCL_TRY        if constexpr ([[maybe_unused]] _CUDA_VSTD::__cccl_catch_any_lvalue __catch_any_lvalue_obj{}; true)
#  define _CCCL_CATCH(...) else if constexpr (false) for (__VA_ARGS__ = __catch_any_lvalue_obj; false;)
#  define _CCCL_CATCH_ALL  else
#  define _CCCL_THROW(...) _CUDA_VSTD_NOVERSION::terminate()
#  define _CCCL_RETHROW    _CUDA_VSTD_NOVERSION::terminate()
#endif // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^

#endif // _LIBCUDACXX___EXCEPTION_EXCEPTION_MACROS_H

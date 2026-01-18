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

#include <cuda/std/__exception/format_error.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__utility/typeid.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdio>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __cccl_catch_any_lvalue
{
  template <class _Tp>
  _CCCL_HOST_DEVICE operator _Tp&() const noexcept;
};

_CCCL_END_NAMESPACE_CUDA_STD

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
//   - the catch clause must always bind to a named variable

// Expand to keywords only for host code when exceptions are enabled. nvc++ in CUDA mode traps when an exception is
// thrown in device code.
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
#  define _CCCL_TRY       try
#  define _CCCL_CATCH     catch
#  define _CCCL_CATCH_ALL catch (...)
#  define _CCCL_CATCH_FALLTHROUGH
#  define _CCCL_THROW(...) throw __VA_ARGS__
#  define _CCCL_RETHROW    throw
#else // ^^^ use exceptions ^^^ / vvv no exceptions vvv
#  define _CCCL_TRY     \
    if constexpr (true) \
    {
#  define _CCCL_CATCH(...)                                                                       \
    }                                                                                            \
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
#  define _CCCL_THROW(...)                                                                                         \
    do                                                                                                             \
    {                                                                                                              \
      NV_IF_TARGET(NV_IS_HOST, ({                                                                                  \
                     ::cuda::__msg_storage __msg_buffer{};                                                         \
                     ::cuda::__format_error(                                                                       \
                       __msg_buffer, ::cuda::std::__pretty_nameof<decltype(__VA_ARGS__)>(), (__VA_ARGS__).what()); \
                     ::fprintf(stderr, "%s\n", __msg_buffer.__buffer);                                             \
                     ::fflush(stderr);                                                                             \
                   }));                                                                                            \
      ::cuda::std::terminate();                                                                                    \
    } while (0)
#  define _CCCL_RETHROW ::cuda::std::terminate()
#endif // ^^^ no exceptions ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_EXCEPTION_MACROS_H

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
#include <cuda/std/__host_stdlib/cstdio>
#include <cuda/std/__host_stdlib/exception>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __cccl_catch_any_lvalue
{
  template <class _Tp>
  _CCCL_API operator _Tp&() const noexcept;
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
//
// Nested exceptions follow the same pattern:
//   _CCCL_THROW_WITH_NESTED(X, ...)  replaces  std::throw_with_nested(X(...)): throws an X that carries the
//                                    active exception as its nested cause (use inside a catch clause)
//   _CCCL_RETHROW_IF_NESTED(e)       replaces  std::rethrow_if_nested(e): rethrows the cause nested in `e`
//                                    if there is one, and has no effect otherwise
//   _CCCL_THROW_MAYBE_WITH_NESTED(X, ...)  throws an X with the active exception nested when there is one
//                                    (inside a handler), and a plain X otherwise. std::throw_with_nested
//                                    outside a handler would store an empty cause, and rethrowing that
//                                    cause later terminates; this form is safe in both places.

// Expand to keywords only for host code when exceptions are enabled. nvc++ in CUDA mode traps when an exception is
// thrown in device code.
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
#  define _CCCL_TRY       try
#  define _CCCL_CATCH     catch
#  define _CCCL_CATCH_ALL catch (...)
#  define _CCCL_CATCH_FALLTHROUGH

// Even though nvc++ in CUDA mode replaces `throw` by `__trap()` call in device code, it instantiates the exception type
// which can introduce some host only symbols to the nvvm ir (for example snprintf). So we need to wrap it by the
// NV_IF_ELSE_TARGET macro.
#  define _CCCL_THROW(_TYPE, ...)                                                             \
    do                                                                                        \
    {                                                                                         \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (throw _TYPE(__VA_ARGS__);), (::cuda::std::terminate();)) \
    } while (0)
#  define _CCCL_RETHROW throw

// std::throw_with_nested and std::rethrow_if_nested are host-only library functions. Device code has no
// exceptions, so no exception is ever nested there: throwing terminates, as _CCCL_THROW does, and
// rethrow-if-nested is correctly a no-op.
#  define _CCCL_THROW_WITH_NESTED(_TYPE, ...)                                                                     \
    do                                                                                                            \
    {                                                                                                             \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (::std::throw_with_nested(_TYPE(__VA_ARGS__));), (::cuda::std::terminate();)) \
    } while (0)
#  define _CCCL_RETHROW_IF_NESTED(...)                                                                 \
    do                                                                                                 \
    {                                                                                                  \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (::std::rethrow_if_nested(__VA_ARGS__);), ((void) (__VA_ARGS__);)) \
    } while (0)
#  define _CCCL_THROW_MAYBE_WITH_NESTED(_TYPE, ...)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
      NV_IF_ELSE_TARGET(                                                                                               \
        NV_IS_HOST,                                                                                                    \
        (if (::std::current_exception()) { ::std::throw_with_nested(_TYPE(__VA_ARGS__)); } throw _TYPE(__VA_ARGS__);), \
        (::cuda::std::terminate();))                                                                                   \
    } while (0)
#else // ^^^ use exceptions ^^^ / vvv no exceptions vvv
#  define _CCCL_TRY     \
    if constexpr (true) \
    {
#  define _CCCL_CATCH(...)    \
    }                         \
    else if constexpr (false) \
    {                         \
      for (__VA_ARGS__ = ::cuda::std::__cccl_catch_any_lvalue{}; false;)
#  define _CCCL_CATCH_ALL \
    }                     \
    else
#  define _CCCL_CATCH_FALLTHROUGH \
    }                             \
    else                          \
    {                             \
    }

#  if _CCCL_HOSTJIT()
#    define _CCCL_THROW(_TYPE, ...)                                              \
      do                                                                         \
      {                                                                          \
        _CCCL_ASSERT(false, "An instance of class " #_TYPE " would be thrown."); \
        ::cuda::std::terminate();                                                \
      } while (0)
#  else // ^^^ _CCCL_HOSTJIT() ^^^ / vvv !_CCCL_HOSTJIT() vvv
#    define _CCCL_THROW(_TYPE, ...)                                                                                \
      do                                                                                                           \
      {                                                                                                            \
        NV_IF_ELSE_TARGET(NV_IS_HOST,                                                                              \
                          ({                                                                                       \
                            ::fprintf(stderr,                                                                      \
                                      "%s:%u: An instance of class %s would be thrown.\n  what():  %s\nAborted\n", \
                                      __FILE__,                                                                    \
                                      __LINE__,                                                                    \
                                      #_TYPE,                                                                      \
                                      (_TYPE(__VA_ARGS__)).what());                                                \
                            ::fflush(stderr);                                                                      \
                          }),                                                                                      \
                          ({ _CCCL_ASSERT(false, "An instance of class " #_TYPE " would be thrown."); }))          \
        ::cuda::std::terminate();                                                                                  \
      } while (0)
#  endif // !_CCCL_HOSTJIT()
#  define _CCCL_RETHROW                             ::cuda::std::terminate()

// Without exceptions there is no active exception to nest, so throwing with a nested cause reports and
// terminates exactly like _CCCL_THROW, and no exception can carry a nested cause, so rethrow-if-nested is
// a no-op (the operand is still evaluated, as std::rethrow_if_nested would).
#  define _CCCL_THROW_WITH_NESTED(_TYPE, ...)       _CCCL_THROW(_TYPE, __VA_ARGS__)
#  define _CCCL_THROW_MAYBE_WITH_NESTED(_TYPE, ...) _CCCL_THROW(_TYPE, __VA_ARGS__)
#  define _CCCL_RETHROW_IF_NESTED(...)              ((void) (__VA_ARGS__))
#endif // ^^^ no exceptions ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_EXCEPTION_MACROS_H

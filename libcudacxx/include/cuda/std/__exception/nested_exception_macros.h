//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXCEPTION_NESTED_EXCEPTION_MACROS_H
#define _CUDA_STD___EXCEPTION_NESTED_EXCEPTION_MACROS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__exception/terminate.h> // IWYU pragma: keep
#include <cuda/std/__host_stdlib/exception> // IWYU pragma: keep

#include <nv/target>

// Nested exceptions follow the same pattern as the macros in exception_macros.h. They live in their own header
// because they need <exception>, which is too heavy for everything that includes exception_macros.h.
//   _CCCL_THROW_WITH_NESTED(X, ...)  replaces  std::throw_with_nested(X(...)): throws an X that carries the
//                                    active exception as its nested cause (use inside a catch clause)
//   _CCCL_RETHROW_IF_NESTED(e)       replaces  std::rethrow_if_nested(e): rethrows the cause nested in `e`
//                                    if there is one, and has no effect otherwise
//   _CCCL_THROW_MAYBE_WITH_NESTED(X, ...)  throws an X with the active exception nested when there is one
//                                    (inside a handler), and a plain X otherwise. std::throw_with_nested
//                                    outside a handler would store an empty cause, and rethrowing that
//                                    cause later terminates; this form is safe in both places.

#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
#  define _CCCL_THROW_WITH_NESTED(_TYPE, ...)                                                                     \
    do                                                                                                            \
    {                                                                                                             \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (::std::throw_with_nested(_TYPE(__VA_ARGS__));), (::cuda::std::terminate();)) \
    } while (0)
#  define _CCCL_RETHROW_IF_NESTED(...)                                                                     \
    do                                                                                                     \
    {                                                                                                      \
      NV_IF_ELSE_TARGET(NV_IS_HOST, (::std::rethrow_if_nested(__VA_ARGS__);), (::cuda::std::terminate();)) \
    } while (0)
#  define _CCCL_THROW_MAYBE_WITH_NESTED(_TYPE, ...)                                                \
    do                                                                                             \
    {                                                                                              \
      NV_IF_ELSE_TARGET(                                                                           \
        NV_IS_HOST,                                                                                \
        (                                                                                          \
          if (::std::current_exception()) { ::std::throw_with_nested(_TYPE(__VA_ARGS__)); } else { \
            throw _TYPE(__VA_ARGS__);                                                              \
          }),                                                                                      \
        (::cuda::std::terminate();))                                                               \
    } while (0)
#else // ^^^ use exceptions ^^^ / vvv no exceptions vvv
#  define _CCCL_THROW_WITH_NESTED(_TYPE, ...)       _CCCL_THROW(_TYPE, __VA_ARGS__)
#  define _CCCL_THROW_MAYBE_WITH_NESTED(_TYPE, ...) _CCCL_THROW(_TYPE, __VA_ARGS__)
#  define _CCCL_RETHROW_IF_NESTED(...)              ::cuda::std::terminate()
#endif // ^^^ no exceptions ^^^

#endif // _CUDA_STD___EXCEPTION_NESTED_EXCEPTION_MACROS_H

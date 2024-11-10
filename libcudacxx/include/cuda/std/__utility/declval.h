//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_DECLVAL_H
#define _LIBCUDACXX___UTILITY_DECLVAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/void_t.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// When variable templates are available, a faster implementation of declval
// is available. It compiles approximately 3x faster than the fallback.
#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)

// NVCC wants the primary variable template to be well-formed. It cannot
// handle the form below which has _Tp&& in the primary template because
// it thinks declval<void>() is ill-formed.
#  if defined(_CCCL_CUDA_COMPILER_NVCC)

template <class _Tp, class = void>
extern void (*declval)() noexcept;

template <class _Tp>
  extern _Tp && (*declval<_Tp, void_t<_Tp&&>>) () noexcept;

#  else // ^^^ defined(_CCCL_CUDA_COMPILER_NVCC) ^^^ / vvv !defined(_CCCL_CUDA_COMPILER_NVCC) vvv

template <class _Tp>
  extern _Tp && (*declval)() noexcept;

template <>
constexpr void (*declval<void>)() noexcept = nullptr; // NOLINT (clang-tidymisc-definitions-in-headers)

template <>
constexpr void (*declval<void const>)() noexcept = nullptr; // NOLINT (clang-tidymisc-definitions-in-headers)

template <>
constexpr void (*declval<void volatile>)() noexcept = nullptr; // NOLINT (clang-tidymisc-definitions-in-headers)

template <>
constexpr void (*declval<void const volatile>)() noexcept = nullptr; // NOLINT (clang-tidymisc-definitions-in-headers)

#  endif // _CCCL_CUDA_COMPILER_NVCC

#else // ^^^ !_CCCL_NO_VARIABLE_TEMPLATES ^^^ / vvv _CCCL_NO_VARIABLE_TEMPLATES vvv

// Suppress deprecation notice for volatile-qualified return type resulting
// from volatile-qualified types _Tp.
_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI _Tp&& __declval(int);
template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI _Tp __declval(long);
_CCCL_SUPPRESS_DEPRECATED_POP

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI decltype(_CUDA_VSTD::__declval<_Tp>(0)) declval() noexcept;

#endif // _CCCL_NO_VARIABLE_TEMPLATES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_DECLVAL_H

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

// When variable templates and noexcept function types are available, a faster
// implementation of declval is available. It compiles approximately 2x faster
// than the fallback. NOTE: this implementation causes nvcc < 12.4 to ICE and
// MSVC < 19.39 to miscompile so we use the fallback instead. The use of the
// `__identity_t` alias is help MSVC parse the declaration correctly.
#if !defined(_CCCL_NO_VARIABLE_TEMPLATES) && !defined(_CCCL_NO_NOEXCEPT_FUNCTION_TYPE) \
  && !_CCCL_CUDA_COMPILER(NVCC, <, 12, 4) && !_CCCL_COMPILER(MSVC, <, 19, 39)

template <class _Tp>
using __identity_t _CCCL_NODEBUG_ALIAS = _Tp;

template <class _Tp, class = void>
extern __identity_t<void (*)() noexcept> declval;

template <class _Tp>
extern __identity_t<_Tp && (*) () noexcept> declval<_Tp, void_t<_Tp&&>>;

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

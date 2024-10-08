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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

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

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_DECLVAL_H

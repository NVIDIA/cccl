//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_CHAR_TRAITS_H
#define _CUDA_STD___FWD_CHAR_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

template <class _CharT>
struct char_traits;

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

// cuda::std:: forward declarations

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT>
struct _CCCL_TYPE_VISIBILITY_DEFAULT char_traits;
template <>
struct char_traits<char>;
#if _CCCL_HAS_CHAR8_T()
template <>
struct char_traits<char8_t>;
#endif // _CCCL_HAS_CHAR8_T()
template <>
struct char_traits<char16_t>;
template <>
struct char_traits<char32_t>;
#if _CCCL_HAS_WCHAR_T()
template <>
struct char_traits<wchar_t>;
#endif // _CCCL_HAS_WCHAR_T()

template <class _Tp>
inline constexpr bool __is_std_char_traits_v = false;
#if _CCCL_HAS_HOST_STD_LIB()
template <class _Tp>
inline constexpr bool __is_std_char_traits_v<::std::char_traits<_Tp>> = true;
#endif // _CCCL_HAS_HOST_STD_LIB()

template <class _Tp>
inline constexpr bool __is_cuda_std_char_traits_v = false;
template <class _Tp>
inline constexpr bool __is_cuda_std_char_traits_v<char_traits<_Tp>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_CHAR_TRAITS_H

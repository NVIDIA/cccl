//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_STRING_VIEW_H
#define _CUDA_STD___FWD_STRING_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/char_traits.h>

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_string_view >= 201606L
_CCCL_BEGIN_NAMESPACE_STD

template <class _CharT, class _Traits>
class basic_string_view;

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_string_view >= 201606L

// cuda::std:: forward declarations

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT, class _Traits = char_traits<_CharT>>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_string_view;

using string_view = basic_string_view<char>;
#if _CCCL_HAS_CHAR8_T()
using u8string_view = basic_string_view<char8_t>;
#endif // _CCCL_HAS_CHAR8_T()
using u16string_view = basic_string_view<char16_t>;
using u32string_view = basic_string_view<char32_t>;
#if _CCCL_HAS_WCHAR_T()
using wstring_view = basic_string_view<wchar_t>;
#endif // _CCCL_HAS_WCHAR_T()

// clang-format off
template <class _CharT, class _Traits>
class _CCCL_PREFERRED_NAME(string_view)
#if _CCCL_HAS_CHAR8_T()
_CCCL_PREFERRED_NAME(u8string_view)
#endif // _CCCL_HAS_CHAR8_T()
_CCCL_PREFERRED_NAME(u16string_view)
_CCCL_PREFERRED_NAME(u32string_view)
#if _CCCL_HAS_WCHAR_T()
_CCCL_PREFERRED_NAME(wstring_view)
#endif // _CCCL_HAS_WCHAR_T()
      basic_string_view;
// clang-format on

template <class _Tp>
inline constexpr bool __is_std_basic_string_view_v = false;
#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_string_view >= 201606L
template <class _CharT, class _Traits>
inline constexpr bool __is_std_basic_string_view_v<::std::basic_string_view<_CharT, _Traits>> = true;
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_string_view >= 201606L

template <class _Tp>
inline constexpr bool __is_cuda_std_basic_string_view_v = false;
template <class _Tp>
inline constexpr bool __is_cuda_std_basic_string_view_v<const _Tp> = __is_cuda_std_basic_string_view_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_basic_string_view_v<volatile _Tp> = __is_cuda_std_basic_string_view_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_basic_string_view_v<const volatile _Tp> = __is_cuda_std_basic_string_view_v<_Tp>;
template <class _CharT, class _Traits>
inline constexpr bool __is_cuda_std_basic_string_view_v<basic_string_view<_CharT, _Traits>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_STRING_VIEW_H

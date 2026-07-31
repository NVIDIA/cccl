//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___STRING_LITERAL_H
#define _CUDA_STD___STRING_LITERAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _CharT, class _CharStr, class _WCharStr, class _Char8Str = void, class _Char16Str, class _Char32Str>
[[nodiscard]] _CCCL_API constexpr const auto& __cccl_select_strlit(
  [[maybe_unused]] const _CharStr(&__char_str),
  [[maybe_unused]] const _WCharStr(&__wchar_str),
#if _CCCL_HAS_CHAR8_T()
  [[maybe_unused]] const _Char8Str(&__char8_str),
#endif // _CCCL_HAS_CHAR8_T()
  [[maybe_unused]] const _Char16Str(&__char16_str),
  [[maybe_unused]] const _Char32Str(&__char32_str)) noexcept
{
  if constexpr (cuda::std::is_same_v<_CharT, char>)
  {
    return __char_str;
  }
  else if constexpr (cuda::std::is_same_v<_CharT, wchar_t>)
  {
    return __wchar_str;
  }
#if _CCCL_HAS_CHAR8_T()
  else if constexpr (cuda::std::is_same_v<_CharT, char8_t>)
  {
    return __char8_str;
  }
#endif // _CCCL_HAS_CHAR8_T()
  else if constexpr (cuda::std::is_same_v<_CharT, char16_t>)
  {
    return __char16_str;
  }
  else if constexpr (cuda::std::is_same_v<_CharT, char32_t>)
  {
    return __char32_str;
  }
  else
  {
    static_assert(cuda::std::__always_false_v<_CharT>,
                  "Unsupported character type. Supported types are char, wchar_t, char8_t, char16_t, and char32_t.");
    _CCCL_UNREACHABLE();
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CHAR8_T()
#  define _CCCL_STRLIT(_TYPE, _STR) ::cuda::std::__cccl_select_strlit<_TYPE>(_STR, L##_STR, u8##_STR, u##_STR, U##_STR)
#else // ^^^ _CCCL_HAS_CHAR8_T() ^^^ / vvv !_CCCL_HAS_CHAR8_T() vvv
#  define _CCCL_STRLIT(_TYPE, _STR) ::cuda::std::__cccl_select_strlit<_TYPE>(_STR, L##_STR, u##_STR, U##_STR)
#endif // ^^^ !_CCCL_HAS_CHAR8_T() ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___STRING_LITERAL_H

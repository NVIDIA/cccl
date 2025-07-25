//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_LITERAL_H
#define TEST_SUPPORT_LITERAL_H

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

template <class CharT>
[[nodiscard]] __host__ __device__ constexpr CharT _test_charlit_impl(
  [[maybe_unused]] char char_val,
  [[maybe_unused]] wchar_t wchar_val,
#if _CCCL_HAS_CHAR8_T()
  [[maybe_unused]] char8_t char8_val,
#endif // _CCCL_HAS_CHAR8_T()
  [[maybe_unused]] char16_t char16_val,
  [[maybe_unused]] char32_t char32_val) noexcept
{
  if constexpr (cuda::std::is_same_v<CharT, char>)
  {
    return char_val;
  }
  else if constexpr (cuda::std::is_same_v<CharT, wchar_t>)
  {
    return wchar_val;
  }
#if _CCCL_HAS_CHAR8_T()
  else if constexpr (cuda::std::is_same_v<CharT, char8_t>)
  {
    return char8_val;
  }
#endif // _CCCL_HAS_CHAR8_T()
  else if constexpr (cuda::std::is_same_v<CharT, char16_t>)
  {
    return char16_val;
  }
  else if constexpr (cuda::std::is_same_v<CharT, char32_t>)
  {
    return char32_val;
  }
  else
  {
    static_assert(cuda::std::__always_false_v<CharT>,
                  "Unsupported character type. Supported types are char, wchar_t, char8_t, char16_t, and char32_t.");
  }
  _CCCL_UNREACHABLE();
}

#if _CCCL_HAS_CHAR8_T()
#  define TEST_CHARLIT(CharT, val) _test_charlit_impl<CharT>(val, L##val, u8##val, u##val, U##val)
#else // _CCCL_HAS_CHAR8_T()
#  define TEST_CHARLIT(CharT, val) _test_charlit_impl<CharT>(val, L##val, u##val, U##val)
#endif // _CCCL_HAS_CHAR8_T()

template <class CharT, cuda::std::size_t N>
[[nodiscard]] __host__ __device__ constexpr auto _test_strlit_impl(
  [[maybe_unused]] const char (&char_str)[N],
  [[maybe_unused]] const wchar_t (&wchar_str)[N],
#if _CCCL_HAS_CHAR8_T()
  [[maybe_unused]] const char8_t (&char8_str)[N],
#endif // _CCCL_HAS_CHAR8_T()
  [[maybe_unused]] const char16_t (&char16_str)[N],
  [[maybe_unused]] const char32_t (&char32_str)[N]) noexcept -> const CharT (&)[N]
{
  if constexpr (cuda::std::is_same_v<CharT, char>)
  {
    return char_str;
  }
  else if constexpr (cuda::std::is_same_v<CharT, wchar_t>)
  {
    return wchar_str;
  }
#if _CCCL_HAS_CHAR8_T()
  else if constexpr (cuda::std::is_same_v<CharT, char8_t>)
  {
    return char8_str;
  }
#endif // _CCCL_HAS_CHAR8_T()
  else if constexpr (cuda::std::is_same_v<CharT, char16_t>)
  {
    return char16_str;
  }
  else if constexpr (cuda::std::is_same_v<CharT, char32_t>)
  {
    return char32_str;
  }
  else
  {
    static_assert(cuda::std::__always_false_v<CharT>,
                  "Unsupported character type. Supported types are char, wchar_t, char8_t, char16_t, and char32_t.");
  }
  _CCCL_UNREACHABLE();
}

#if _CCCL_HAS_CHAR8_T()
#  define TEST_STRLIT(CharT, str) _test_strlit_impl<CharT>(str, L##str, u8##str, u##str, U##str)
#else // _CCCL_HAS_CHAR8_T()
#  define TEST_STRLIT(CharT, str) _test_strlit_impl<CharT>(str, L##str, u##str, U##str)
#endif // _CCCL_HAS_CHAR8_T()

#endif // TEST_SUPPORT_LITERAL_H

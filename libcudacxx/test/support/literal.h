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

template <class T>
struct _test_int_literal_impl_result
{
  T value;
  bool invalid_character;
  bool overflow;
};

template <int Base>
[[nodiscard]] __host__ __device__ constexpr bool _test_int_literal_char_to_digit(char c, int& value)
{
  static_assert(Base >= 2 && Base <= 36, "Base must be between 2 and 36 inclusive.");

  if constexpr (Base <= 10)
  {
    if (c >= '0' && c < '0' + Base)
    {
      value = c - '0';
      return true;
    }
    return false;
  }
  else
  {
    if (c >= '0' && c < '0' + 10)
    {
      value = c - '0';
      return true;
    }
    else if (c >= 'A' && c < 'A' + (Base - 10))
    {
      value = c - 'A' + 10;
      return true;
    }
    else if (c >= 'a' && c < 'a' + (Base - 10))
    {
      value = c - 'a' + 10;
      return true;
    }
    return false;
  }
}

template <class T, unsigned Base>
[[nodiscard]] __host__ __device__ constexpr _test_int_literal_impl_result<T>
_test_int_literal_impl(const char* begin, const char* end) noexcept
{
  using U         = cuda::std::make_unsigned_t<T>;
  constexpr U max = (~U{0}) >> cuda::std::is_signed_v<T>;

  _test_int_literal_impl_result<T> result{};

  U value = 0;

  const char* it = begin;
  for (; it != end; ++it)
  {
    if (*it == '\'')
    {
      continue;
    }

    int digit{};
    if (!_test_int_literal_char_to_digit<Base>(*it, digit))
    {
      result.invalid_character = true;
      return result;
    }

    const U new_value = value * Base + digit;
    if (new_value < value || new_value > max)
    {
      result.overflow = true;
      return result;
    }
    value = new_value;
  }

  result.value = static_cast<T>(value);
  return result;
}

template <class T, class SizeT = decltype(sizeof(int)), SizeT N>
[[nodiscard]] __host__ __device__ constexpr _test_int_literal_impl_result<T>
_test_int_literal_impl(const char (&cs)[N]) noexcept
{
  unsigned base = 10;
  SizeT offset  = 0;

  if (N >= 2 && cs[0] == '0')
  {
    if (cs[1] == 'b' || cs[1] == 'B')
    {
      base   = 2;
      offset = 2;
    }
    else if (cs[1] == 'x' || cs[1] == 'X')
    {
      base   = 16;
      offset = 2;
    }
    else
    {
      for (SizeT i = 1; i < N; ++i)
      {
        base   = 8;
        offset = 1;

        if (!(cs[i] >= '0' && cs[i] <= '7') && cs[i] != '\'')
        {
          base   = 10;
          offset = 0;
          break;
        }
      }
    }
  }

  switch (base)
  {
    case 2:
      return _test_int_literal_impl<T, 2>(cs + offset, cs + N);
    case 8:
      return _test_int_literal_impl<T, 8>(cs + offset, cs + N);
    case 16:
      return _test_int_literal_impl<T, 16>(cs + offset, cs + N);
    case 10:
    default:
      return _test_int_literal_impl<T, 10>(cs + offset, cs + N);
  }
}

namespace test_integer_literals
{

#if _CCCL_HAS_INT128()
template <char... Cs>
[[nodiscard]] __host__ __device__ constexpr __int128_t operator""_i128() noexcept
{
  constexpr char cs[]{Cs...};
  constexpr auto result = _test_int_literal_impl<__int128_t>(cs);
  static_assert(!result.invalid_character, "Invalid character in integer literal.");
  static_assert(!result.overflow, "Integer literal overflow.");
  return result.value;
}

template <char... Cs>
[[nodiscard]] __host__ __device__ constexpr __uint128_t operator""_u128() noexcept
{
  constexpr char cs[]{Cs...};
  constexpr auto result = _test_int_literal_impl<__uint128_t>(cs);
  static_assert(!result.invalid_character, "Invalid character in integer literal.");
  static_assert(!result.overflow, "Integer literal overflow.");
  return result.value;
}
#endif // _CCCL_HAS_INT128()

} // namespace test_integer_literals

#endif // TEST_SUPPORT_LITERAL_H

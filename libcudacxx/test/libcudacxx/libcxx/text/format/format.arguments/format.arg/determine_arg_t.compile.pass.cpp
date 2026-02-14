//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// cuda::std::__fmt_determine_arg_t

#include <cuda/std/__format_>
#include <cuda/std/__string_>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

struct FormattableType
{};
struct UnformattableType
{};

template <class CharT>
struct cuda::std::formatter<FormattableType, CharT>
{
  template <class ParseContext>
  __host__ __device__ constexpr typename ParseContext::iterator parse(ParseContext& pc);

  template <class FmtContext>
  __host__ __device__ typename FmtContext::iterator format(FormattableType v, FmtContext& ctx) const;
};

template <class CharT>
__host__ __device__ void test_arg_of_v()
{
  using cuda::std::__fmt_arg_t;
  using cuda::std::__fmt_determine_arg_t;

  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  // Boolean
  static_assert(__fmt_determine_arg_t<Context, bool>() == __fmt_arg_t::__boolean);

  // Char types
  static_assert(__fmt_determine_arg_t<Context, CharT>() == __fmt_arg_t::__char_type);
#if _CCCL_HAS_WCHAR_T()
  if constexpr (cuda::std::is_same_v<CharT, wchar_t>)
  {
    static_assert(__fmt_determine_arg_t<Context, char>() == __fmt_arg_t::__char_type);
  }
#endif // _CCCL_HAS_WCHAR_T()

  // Signed integer types
  static_assert(__fmt_determine_arg_t<Context, signed char>() == __fmt_arg_t::__int);
  static_assert(__fmt_determine_arg_t<Context, signed short>() == __fmt_arg_t::__int);
  static_assert(__fmt_determine_arg_t<Context, signed int>() == __fmt_arg_t::__int);
  static_assert(__fmt_determine_arg_t<Context, signed long>()
                == ((sizeof(signed long) == sizeof(signed int)) ? __fmt_arg_t::__int : __fmt_arg_t::__long_long));
  static_assert(__fmt_determine_arg_t<Context, signed long long>() == __fmt_arg_t::__long_long);

  // Unsigned integer types
  static_assert(__fmt_determine_arg_t<Context, unsigned char>() == __fmt_arg_t::__unsigned);
  static_assert(__fmt_determine_arg_t<Context, unsigned short>() == __fmt_arg_t::__unsigned);
  static_assert(__fmt_determine_arg_t<Context, unsigned int>() == __fmt_arg_t::__unsigned);
  static_assert(
    __fmt_determine_arg_t<Context, unsigned long>()
    == ((sizeof(unsigned long) == sizeof(unsigned int)) ? __fmt_arg_t::__unsigned : __fmt_arg_t::__unsigned_long_long));
  static_assert(__fmt_determine_arg_t<Context, unsigned long long>() == __fmt_arg_t::__unsigned_long_long);

  // Floating-point types
  static_assert(__fmt_determine_arg_t<Context, float>() == __fmt_arg_t::__float);
  static_assert(__fmt_determine_arg_t<Context, double>() == __fmt_arg_t::__double);
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(__fmt_determine_arg_t<Context, long double>() == __fmt_arg_t::__long_double);
#endif // _CCCL_HAS_LONG_DOUBLE()

  // Const char pointer types
  static_assert(__fmt_determine_arg_t<Context, CharT*>() == __fmt_arg_t::__const_char_type_ptr);
  static_assert(__fmt_determine_arg_t<Context, const CharT*>() == __fmt_arg_t::__const_char_type_ptr);

  // String view types
  static_assert(__fmt_determine_arg_t<Context, CharT[10]>() == __fmt_arg_t::__string_view);
  static_assert(__fmt_determine_arg_t<Context, cuda::std::basic_string_view<CharT>>() == __fmt_arg_t::__string_view);
  {
    struct TestCharTraits : cuda::std::char_traits<CharT>
    {};
    static_assert(__fmt_determine_arg_t<Context, cuda::std::basic_string_view<CharT, TestCharTraits>>()
                  == __fmt_arg_t::__string_view);
  }

  // Pointer types
  static_assert(__fmt_determine_arg_t<Context, void*>() == __fmt_arg_t::__ptr);
  static_assert(__fmt_determine_arg_t<Context, const void*>() == __fmt_arg_t::__ptr);
  static_assert(__fmt_determine_arg_t<Context, cuda::std::nullptr_t>() == __fmt_arg_t::__ptr);

  // Handle types
  static_assert(__fmt_determine_arg_t<Context, FormattableType>() == __fmt_arg_t::__handle);
#if _CCCL_HAS_INT128()
  static_assert(__fmt_determine_arg_t<Context, __int128_t>() == __fmt_arg_t::__handle);
  static_assert(__fmt_determine_arg_t<Context, __uint128_t>() == __fmt_arg_t::__handle);
#endif // _CCCL_HAS_INT128()

  // Unknown types
  static_assert(__fmt_determine_arg_t<Context, void>() == __fmt_arg_t::__none);
  static_assert(__fmt_determine_arg_t<Context, CharT[]>() == __fmt_arg_t::__none);
  static_assert(__fmt_determine_arg_t<Context, int[10]>() == __fmt_arg_t::__none);
  static_assert(__fmt_determine_arg_t<Context, UnformattableType>() == __fmt_arg_t::__none);
}

__host__ __device__ void test()
{
  test_arg_of_v<char>();
#if _CCCL_HAS_WCHAR_T()
  test_arg_of_v<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// template<class charT, class traits, class Allocator>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os,
//              const basic_string_view<charT,traits> str);

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <sstream>

#include "literal.h"

template <class CharT>
void test_with_default_type_traits()
{
  using OS = std::basic_ostringstream<CharT>;
  using SV = cuda::std::basic_string_view<CharT>;

  // check that cuda::std::char_traits are mapped to std::char_traits
  static_assert(cuda::std::is_same_v<typename OS::char_type, CharT>);
  static_assert(cuda::std::is_same_v<typename OS::traits_type, std::char_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename SV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename SV::traits_type, cuda::std::char_traits<CharT>>);

  const CharT* str = TEST_STRLIT(CharT, "some text");

  // 1. test basic write without formatting
  {
    OS out{};
    SV sv{str};

    out << sv;
    assert(out.good());
    assert(out.str() == str);
  }

  // 2. test basic write with formatting
  {
    OS out{};
    SV sv{str};

    out.width(12);
    out << sv;
    assert(out.good());
    assert(out.str() == TEST_STRLIT(CharT, "   some text"));
  }
}

template <class CharT>
struct custom_type_traits
    : private std::char_traits<CharT>
    , private cuda::std::char_traits<CharT>
{
  using char_type  = typename cuda::std::char_traits<CharT>::char_type;
  using int_type   = typename cuda::std::char_traits<CharT>::int_type;
  using pos_type   = typename std::char_traits<CharT>::pos_type;
  using off_type   = typename std::char_traits<CharT>::off_type;
  using state_type = typename std::char_traits<CharT>::state_type;

  using cuda::std::char_traits<CharT>::assign;
  using cuda::std::char_traits<CharT>::eq;
  using cuda::std::char_traits<CharT>::lt;
  using cuda::std::char_traits<CharT>::compare;
  using cuda::std::char_traits<CharT>::length;
  using cuda::std::char_traits<CharT>::find;
  using cuda::std::char_traits<CharT>::move;
  using cuda::std::char_traits<CharT>::copy;
  using cuda::std::char_traits<CharT>::to_char_type;
  using cuda::std::char_traits<CharT>::to_int_type;
  using cuda::std::char_traits<CharT>::eq_int_type;
  using std::char_traits<CharT>::eof;
  using std::char_traits<CharT>::not_eof;
};

template <class CharT>
void test_with_custom_type_traits()
{
  using OS = std::basic_ostringstream<CharT, custom_type_traits<CharT>>;
  using SV = cuda::std::basic_string_view<CharT, custom_type_traits<CharT>>;

  // check that cuda::std::char_traits are mapped to std::char_traits
  static_assert(cuda::std::is_same_v<typename OS::char_type, CharT>);
  static_assert(cuda::std::is_same_v<typename OS::traits_type, custom_type_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename SV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename SV::traits_type, custom_type_traits<CharT>>);

  const CharT* str = TEST_STRLIT(CharT, "some text");

  // 1. test basic write without formatting
  {
    OS out{};
    SV sv{str};

    out << sv;
    assert(out.good());
    assert(out.str() == str);
  }

  // 2. test basic write with formatting
  {
    OS out{};
    SV sv{str};

    out.width(12);
    out << sv;
    assert(out.good());
    assert(out.str() == TEST_STRLIT(CharT, "   some text"));
  }
}

template <class CharT>
void test_type()
{
  test_with_default_type_traits<CharT>();
  test_with_custom_type_traits<CharT>();
}

void test()
{
  test_type<char>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}

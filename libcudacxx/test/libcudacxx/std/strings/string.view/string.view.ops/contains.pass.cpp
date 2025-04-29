//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr bool contains(basic_string_view x) const noexcept;

// constexpr bool contains(charT x) const noexcept;

// constexpr bool contains(const charT* x) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_contains()
{
  using CharT = typename SV::value_type;

  // constexpr bool contains(basic_string_view x) const noexcept;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.contains(SV{}))>);
  static_assert(noexcept(SV{}.contains(SV{})));

  {
    const CharT* str = TEST_STRLIT(CharT, "abcde");

    SV a0{};
    SV a1{str + 1, 1};
    SV a3{str + 1, 3};
    SV a5{str};
    SV aNot{TEST_STRLIT(CharT, "xyz")};

    SV b0{};
    SV b1{str + 1, 1};
    SV b2{str + 1, 2};
    SV b3{str + 1, 3};
    SV b4{str + 1, 4};
    SV b5{str};
    SV bNot{TEST_STRLIT(CharT, "xyz")};
    SV bNot2{TEST_STRLIT(CharT, "bd")};
    SV bNot3{TEST_STRLIT(CharT, "dcb")};

    assert(a0.contains(b0));
    assert(!a0.contains(b1));

    assert(a1.contains(b0));
    assert(a1.contains(b1));
    assert(!a1.contains(b2));
    assert(!a1.contains(b3));
    assert(!a1.contains(b4));
    assert(!a1.contains(b5));
    assert(!a1.contains(bNot));
    assert(!a1.contains(bNot2));
    assert(!a1.contains(bNot3));

    assert(a3.contains(b0));
    assert(a3.contains(b1));
    assert(a3.contains(b2));
    assert(a3.contains(b3));
    assert(!a3.contains(b4));
    assert(!a3.contains(b5));
    assert(!a3.contains(bNot));
    assert(!a3.contains(bNot2));
    assert(!a3.contains(bNot3));

    assert(a5.contains(b0));
    assert(a5.contains(b1));
    assert(a5.contains(b2));
    assert(a5.contains(b3));
    assert(a5.contains(b4));
    assert(a5.contains(b5));
    assert(!a5.contains(bNot));
    assert(!a5.contains(bNot2));
    assert(!a5.contains(bNot3));

    assert(aNot.contains(b0));
    assert(!aNot.contains(b1));
    assert(!aNot.contains(b2));
    assert(!aNot.contains(b3));
    assert(!aNot.contains(b4));
    assert(!aNot.contains(b5));
    assert(aNot.contains(bNot));
    assert(!aNot.contains(bNot2));
    assert(!aNot.contains(bNot3));
  }

  // constexpr bool contains(charT x) const noexcept;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.contains(CharT{}))>);
  static_assert(noexcept(SV{}.contains(CharT{})));

  {
    SV sv1{};
    SV sv2{TEST_STRLIT(CharT, "abcde")};

    assert(!sv1.contains('c'));
    assert(!sv1.contains('e'));
    assert(!sv1.contains('x'));
    assert(sv2.contains('c'));
    assert(sv2.contains('e'));
    assert(!sv2.contains('x'));
  }

  // constexpr bool contains(const charT* x) const;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.contains(cuda::std::declval<const CharT*>()))>);
  static_assert(noexcept(SV{}.contains(cuda::std::declval<const CharT*>())));

  {
    const CharT* str = TEST_STRLIT(CharT, "abcde");

    SV sv0{};
    SV sv1{str + 4, 1};
    SV sv2{str + 2, 3};
    SV svNot{TEST_STRLIT(CharT, "xyz")};

    assert(sv0.contains(TEST_STRLIT(CharT, "")));
    assert(!sv0.contains(TEST_STRLIT(CharT, "e")));

    assert(sv1.contains(TEST_STRLIT(CharT, "")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "d")));
    assert(sv1.contains(TEST_STRLIT(CharT, "e")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "de")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "cd")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "cde")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "bcde")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "abcde")));
    assert(!sv1.contains(TEST_STRLIT(CharT, "xyz")));

    assert(sv2.contains(TEST_STRLIT(CharT, "")));
    assert(sv2.contains(TEST_STRLIT(CharT, "d")));
    assert(sv2.contains(TEST_STRLIT(CharT, "e")));
    assert(sv2.contains(TEST_STRLIT(CharT, "de")));
    assert(sv2.contains(TEST_STRLIT(CharT, "cd")));
    assert(!sv2.contains(TEST_STRLIT(CharT, "ce")));
    assert(sv2.contains(TEST_STRLIT(CharT, "cde")));
    assert(!sv2.contains(TEST_STRLIT(CharT, "edc")));
    assert(!sv2.contains(TEST_STRLIT(CharT, "bcde")));
    assert(!sv2.contains(TEST_STRLIT(CharT, "abcde")));
    assert(!sv2.contains(TEST_STRLIT(CharT, "xyz")));

    assert(svNot.contains(TEST_STRLIT(CharT, "")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "d")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "e")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "de")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "cd")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "cde")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "bcde")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "abcde")));
    assert(svNot.contains(TEST_STRLIT(CharT, "xyz")));
    assert(!svNot.contains(TEST_STRLIT(CharT, "zyx")));
  }
}

__host__ __device__ constexpr bool test()
{
  test_contains<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_contains<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_contains<cuda::std::u16string_view>();
  test_contains<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_contains<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

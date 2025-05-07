//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr bool starts_with(basic_string_view x) const noexcept;

// constexpr bool starts_with(charT x) const noexcept;

// constexpr bool starts_with(const charT* x) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_starts_with()
{
  using CharT = typename SV::value_type;

  // constexpr bool starts_with(basic_string_view x) const noexcept;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.starts_with(SV{}))>);
  static_assert(noexcept(SV{}.starts_with(SV{})));

  {
    const CharT* str = TEST_STRLIT(CharT, "abcde");

    SV a0{};
    SV a1{str, 1};
    SV a2{str, 2};
    SV aNot{TEST_STRLIT(CharT, "def")};

    SV b0{};
    SV b1{str, 1};
    SV b2{str, 2};
    SV b3{str, 3};
    SV b4{str, 4};
    SV b5{str, 5};
    SV bNot{TEST_STRLIT(CharT, "def")};

    assert(a0.starts_with(b0));
    assert(!a0.starts_with(b1));

    assert(a1.starts_with(b0));
    assert(a1.starts_with(b1));
    assert(!a1.starts_with(b2));
    assert(!a1.starts_with(b3));
    assert(!a1.starts_with(b4));
    assert(!a1.starts_with(b5));
    assert(!a1.starts_with(bNot));

    assert(a2.starts_with(b0));
    assert(a2.starts_with(b1));
    assert(a2.starts_with(b2));
    assert(!a2.starts_with(b3));
    assert(!a2.starts_with(b4));
    assert(!a2.starts_with(b5));
    assert(!a2.starts_with(bNot));

    assert(aNot.starts_with(b0));
    assert(!aNot.starts_with(b1));
    assert(!aNot.starts_with(b2));
    assert(!aNot.starts_with(b3));
    assert(!aNot.starts_with(b4));
    assert(!aNot.starts_with(b5));
    assert(aNot.starts_with(bNot));
  }

  // constexpr bool starts_with(charT x) const noexcept;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.starts_with(CharT{}))>);
  static_assert(noexcept(SV{}.starts_with(CharT{})));

  {
    SV sv{};
    assert(!sv.starts_with('a'));
    assert(!sv.starts_with('x'));
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcde")};
    assert(sv.starts_with('a'));
    assert(!sv.starts_with('x'));
  }

  // constexpr bool starts_with(const charT* x) const;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.starts_with(cuda::std::declval<const CharT*>()))>);
  static_assert(noexcept(SV{}.starts_with(cuda::std::declval<const CharT*>())));

  {
    const CharT* str = TEST_STRLIT(CharT, "abcde");
    SV sv0{};
    SV sv1{str, 1};
    SV sv2{str, 2};
    SV svNot{TEST_STRLIT(CharT, "def")};

    assert(sv0.starts_with(TEST_STRLIT(CharT, "")));
    assert(!sv0.starts_with(TEST_STRLIT(CharT, "a")));

    assert(sv1.starts_with(TEST_STRLIT(CharT, "")));
    assert(sv1.starts_with(TEST_STRLIT(CharT, "a")));
    assert(!sv1.starts_with(TEST_STRLIT(CharT, "ab")));
    assert(!sv1.starts_with(TEST_STRLIT(CharT, "abc")));
    assert(!sv1.starts_with(TEST_STRLIT(CharT, "abcd")));
    assert(!sv1.starts_with(TEST_STRLIT(CharT, "abcde")));
    assert(!sv1.starts_with(TEST_STRLIT(CharT, "def")));

    assert(sv2.starts_with(TEST_STRLIT(CharT, "")));
    assert(sv2.starts_with(TEST_STRLIT(CharT, "a")));
    assert(sv2.starts_with(TEST_STRLIT(CharT, "ab")));
    assert(!sv2.starts_with(TEST_STRLIT(CharT, "abc")));
    assert(!sv2.starts_with(TEST_STRLIT(CharT, "abcd")));
    assert(!sv2.starts_with(TEST_STRLIT(CharT, "abcde")));
    assert(!sv2.starts_with(TEST_STRLIT(CharT, "def")));

    assert(svNot.starts_with(TEST_STRLIT(CharT, "")));
    assert(!svNot.starts_with(TEST_STRLIT(CharT, "a")));
    assert(!svNot.starts_with(TEST_STRLIT(CharT, "ab")));
    assert(!svNot.starts_with(TEST_STRLIT(CharT, "abc")));
    assert(!svNot.starts_with(TEST_STRLIT(CharT, "abcd")));
    assert(!svNot.starts_with(TEST_STRLIT(CharT, "abcde")));
    assert(svNot.starts_with(TEST_STRLIT(CharT, "def")));
  }
}

__host__ __device__ constexpr bool test()
{
  test_starts_with<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_starts_with<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_starts_with<cuda::std::u16string_view>();
  test_starts_with<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_starts_with<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

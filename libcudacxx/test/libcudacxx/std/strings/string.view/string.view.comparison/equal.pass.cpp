//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// template<class charT, class traits>
//   constexpr bool operator==(basic_string_view<charT, traits> lhs, basic_string_view<charT, traits> rhs);
// (plus "sufficient additional overloads" to make implicit conversions work as intended)

#include <cuda/std/cassert>
#include <cuda/std/string_view>

#include "literal.h"
#include "test_macros.h"

template <class T>
struct ConvertibleTo
{
  T t_;
  __host__ __device__ constexpr explicit ConvertibleTo(T t)
      : t_(t)
  {}
  __host__ __device__ constexpr operator T() const
  {
    return t_;
  }
};

template <class SV>
__host__ __device__ constexpr void test_equal()
{
  using CharT = typename SV::value_type;

  static_assert(cuda::std::is_same_v<bool, decltype(operator==(SV{}, SV{}))>);
  static_assert(noexcept(operator==(SV{}, SV{})));

  // Test the behavior of the operator, both with and without implicit conversions.

  {
    SV v[] = {
      TEST_STRLIT(CharT, ""),
      TEST_STRLIT(CharT, "abc"),
      TEST_STRLIT(CharT, "abcdef"),
      TEST_STRLIT(CharT, "acb"),
    };

    for (cuda::std::size_t i = 0; i < cuda::std::size(v); ++i)
    {
      for (cuda::std::size_t j = 0; j < cuda::std::size(v); ++j)
      {
        const bool expected = (i == j);

        assert((v[i] == v[j]) == expected);
        assert((v[i].data() == v[j]) == expected);
        assert((v[i] == v[j].data()) == expected);
        assert((ConvertibleTo<SV>(v[i]) == v[j]) == expected);
        assert((v[i] == ConvertibleTo<SV>(v[j])) == expected);
      }
    }
  }

  // Test its behavior with embedded null bytes.

  {
    SV abc{TEST_STRLIT(CharT, "abc")};
    SV abc0def{TEST_STRLIT(CharT, "abc\0def"), 7};
    SV abcdef{TEST_STRLIT(CharT, "abcdef")};

    assert((abc == abc0def) == false);
    assert((abc == abcdef) == false);
    assert((abc0def == abc) == false);
    assert((abc0def == abcdef) == false);
    assert((abcdef == abc) == false);
    assert((abcdef == abc0def) == false);

    assert((abc.data() == abc0def) == false);
    assert((abc0def == abc.data()) == false);
  }
}

__host__ __device__ constexpr bool test()
{
  test_equal<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_equal<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_equal<cuda::std::u16string_view>();
  test_equal<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_equal<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type find(basic_string_view s, size_type pos = 0) const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_find(const SV& sv, const SV& str, typename SV::size_type x)
{
  assert(sv.find(str) == x);
  if (x != SV::npos)
  {
    assert(x + str.size() <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void
test_find(const SV& sv, const SV& str, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.find(str, pos) == x);
  if (x != SV::npos)
  {
    assert(pos <= x && x + str.size() <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_find()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find(SV{}))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find(SV{}, SizeT{}))>);

  static_assert(noexcept(SV{}.find(SV{})));
  static_assert(noexcept(SV{}.find(SV{}, SizeT{})));

  const SV str1 = TEST_STRLIT(CharT, "");
  const SV str2 = TEST_STRLIT(CharT, "abcde");
  const SV str3 = TEST_STRLIT(CharT, "abcdeabcde");
  const SV str4 = TEST_STRLIT(CharT, "abcdeabcdeabcdeabcde");

  {
    SV sv{str1};
    test_find(sv, str1, 0);
    test_find(sv, str2, SV::npos);
    test_find(sv, str3, SV::npos);
    test_find(sv, str4, SV::npos);
    test_find(sv, str1, 0, 0);
    test_find(sv, str2, 0, SV::npos);
    test_find(sv, str3, 0, SV::npos);
    test_find(sv, str4, 0, SV::npos);
    test_find(sv, str1, 1, SV::npos);
    test_find(sv, str2, 1, SV::npos);
    test_find(sv, str3, 1, SV::npos);
    test_find(sv, str4, 1, SV::npos);
  }
  {
    SV sv{str2};
    test_find(sv, str1, 0);
    test_find(sv, str2, 0);
    test_find(sv, str3, SV::npos);
    test_find(sv, str4, SV::npos);
    test_find(sv, str1, 0, 0);
    test_find(sv, str2, 0, 0);
    test_find(sv, str3, 0, SV::npos);
    test_find(sv, str4, 0, SV::npos);
    test_find(sv, str1, 1, 1);
    test_find(sv, str2, 1, SV::npos);
    test_find(sv, str3, 1, SV::npos);
    test_find(sv, str4, 1, SV::npos);
    test_find(sv, str1, 2, 2);
    test_find(sv, str2, 2, SV::npos);
    test_find(sv, str3, 2, SV::npos);
    test_find(sv, str4, 2, SV::npos);
    test_find(sv, str1, 4, 4);
    test_find(sv, str2, 4, SV::npos);
    test_find(sv, str3, 4, SV::npos);
    test_find(sv, str4, 4, SV::npos);
    test_find(sv, str1, 5, 5);
    test_find(sv, str2, 5, SV::npos);
    test_find(sv, str3, 5, SV::npos);
    test_find(sv, str4, 5, SV::npos);
    test_find(sv, str1, 6, SV::npos);
    test_find(sv, str2, 6, SV::npos);
    test_find(sv, str3, 6, SV::npos);
    test_find(sv, str4, 6, SV::npos);
  }
  {
    SV sv{str3};
    test_find(sv, str1, 0);
    test_find(sv, str2, 0);
    test_find(sv, str3, 0);
    test_find(sv, str4, SV::npos);
    test_find(sv, str1, 0, 0);
    test_find(sv, str2, 0, 0);
    test_find(sv, str3, 0, 0);
    test_find(sv, str4, 0, SV::npos);
    test_find(sv, str1, 1, 1);
    test_find(sv, str2, 1, 5);
    test_find(sv, str3, 1, SV::npos);
    test_find(sv, str4, 1, SV::npos);
    test_find(sv, str1, 5, 5);
    test_find(sv, str2, 5, 5);
    test_find(sv, str3, 5, SV::npos);
    test_find(sv, str4, 5, SV::npos);
    test_find(sv, str1, 9, 9);
    test_find(sv, str2, 9, SV::npos);
    test_find(sv, str3, 9, SV::npos);
    test_find(sv, str4, 9, SV::npos);
    test_find(sv, str1, 10, 10);
    test_find(sv, str2, 10, SV::npos);
    test_find(sv, str3, 10, SV::npos);
    test_find(sv, str4, 10, SV::npos);
    test_find(sv, str1, 11, SV::npos);
    test_find(sv, str2, 11, SV::npos);
    test_find(sv, str3, 11, SV::npos);
    test_find(sv, str4, 11, SV::npos);
  }
  {
    SV sv{str4};
    test_find(sv, str1, 0);
    test_find(sv, str2, 0);
    test_find(sv, str3, 0);
    test_find(sv, str4, 0);
    test_find(sv, str1, 0, 0);
    test_find(sv, str2, 0, 0);
    test_find(sv, str3, 0, 0);
    test_find(sv, str4, 0, 0);
    test_find(sv, str1, 1, 1);
    test_find(sv, str2, 1, 5);
    test_find(sv, str3, 1, 5);
    test_find(sv, str4, 1, SV::npos);
    test_find(sv, str1, 10, 10);
    test_find(sv, str2, 10, 10);
    test_find(sv, str3, 10, 10);
    test_find(sv, str4, 10, SV::npos);
    test_find(sv, str1, 19, 19);
    test_find(sv, str2, 19, SV::npos);
    test_find(sv, str3, 19, SV::npos);
    test_find(sv, str4, 19, SV::npos);
    test_find(sv, str1, 20, 20);
    test_find(sv, str2, 20, SV::npos);
    test_find(sv, str3, 20, SV::npos);
    test_find(sv, str4, 20, SV::npos);
    test_find(sv, str1, 21, SV::npos);
    test_find(sv, str2, 21, SV::npos);
    test_find(sv, str3, 21, SV::npos);
    test_find(sv, str4, 21, SV::npos);
  }
}

__host__ __device__ constexpr bool test()
{
  test_find<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_find<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_find<cuda::std::u16string_view>();
  test_find<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_find<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

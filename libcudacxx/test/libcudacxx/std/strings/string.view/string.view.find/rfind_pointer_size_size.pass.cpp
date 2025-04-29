//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type rfind(const charT* s, size_type pos, size_type n) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_rfind(
  const SV& sv,
  const typename SV::value_type* str,
  typename SV::size_type pos,
  typename SV::size_type n,
  typename SV::size_type x)
{
  assert(sv.rfind(str, pos, n) == x);
  if (x != SV::npos)
  {
    assert(x <= pos && x + n <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_rfind()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(
    cuda::std::is_same_v<SizeT, decltype(SV{}.rfind(cuda::std::declval<const CharT*>(), SizeT{}, SizeT{}))>);
  static_assert(noexcept(SV{}.rfind(cuda::std::declval<const CharT*>(), SizeT{}, SizeT{})));

  const auto str1 = TEST_STRLIT(CharT, "");
  const auto str2 = TEST_STRLIT(CharT, "abcde");
  const auto str3 = TEST_STRLIT(CharT, "abcdeabcde");
  const auto str4 = TEST_STRLIT(CharT, "abcdeabcdeabcdeabcde");

  {
    SV sv{str1};
    test_rfind(sv, str1, 0, 0, 0);
    test_rfind(sv, str2, 0, 0, 0);
    test_rfind(sv, str2, 0, 1, SV::npos);
    test_rfind(sv, str2, 0, 2, SV::npos);
    test_rfind(sv, str2, 0, 4, SV::npos);
    test_rfind(sv, str2, 0, 5, SV::npos);
    test_rfind(sv, str3, 0, 0, 0);
    test_rfind(sv, str3, 0, 1, SV::npos);
    test_rfind(sv, str3, 0, 5, SV::npos);
    test_rfind(sv, str3, 0, 9, SV::npos);
    test_rfind(sv, str3, 0, 10, SV::npos);
    test_rfind(sv, str4, 0, 0, 0);
    test_rfind(sv, str4, 0, 1, SV::npos);
    test_rfind(sv, str4, 0, 10, SV::npos);
    test_rfind(sv, str4, 0, 19, SV::npos);
    test_rfind(sv, str4, 0, 20, SV::npos);
    test_rfind(sv, str1, 1, 0, 0);
    test_rfind(sv, str2, 1, 0, 0);
    test_rfind(sv, str2, 1, 1, SV::npos);
    test_rfind(sv, str2, 1, 2, SV::npos);
    test_rfind(sv, str2, 1, 4, SV::npos);
    test_rfind(sv, str2, 1, 5, SV::npos);
    test_rfind(sv, str3, 1, 0, 0);
    test_rfind(sv, str3, 1, 1, SV::npos);
    test_rfind(sv, str3, 1, 5, SV::npos);
    test_rfind(sv, str3, 1, 9, SV::npos);
    test_rfind(sv, str3, 1, 10, SV::npos);
    test_rfind(sv, str4, 1, 0, 0);
    test_rfind(sv, str4, 1, 1, SV::npos);
    test_rfind(sv, str4, 1, 10, SV::npos);
    test_rfind(sv, str4, 1, 19, SV::npos);
    test_rfind(sv, str4, 1, 20, SV::npos);
  }
  {
    SV sv{str2};
    test_rfind(sv, str1, 0, 0, 0);
    test_rfind(sv, str2, 0, 0, 0);
    test_rfind(sv, str2, 0, 1, 0);
    test_rfind(sv, str2, 0, 2, 0);
    test_rfind(sv, str2, 0, 4, 0);
    test_rfind(sv, str2, 0, 5, 0);
    test_rfind(sv, str3, 0, 0, 0);
    test_rfind(sv, str3, 0, 1, 0);
    test_rfind(sv, str3, 0, 5, 0);
    test_rfind(sv, str3, 0, 9, SV::npos);
    test_rfind(sv, str3, 0, 10, SV::npos);
    test_rfind(sv, str4, 0, 0, 0);
    test_rfind(sv, str4, 0, 1, 0);
    test_rfind(sv, str4, 0, 10, SV::npos);
    test_rfind(sv, str4, 0, 19, SV::npos);
    test_rfind(sv, str4, 0, 20, SV::npos);
    test_rfind(sv, str1, 1, 0, 1);
    test_rfind(sv, str2, 1, 0, 1);
    test_rfind(sv, str2, 1, 1, 0);
    test_rfind(sv, str2, 1, 2, 0);
    test_rfind(sv, str2, 1, 4, 0);
    test_rfind(sv, str2, 1, 5, 0);
    test_rfind(sv, str3, 1, 0, 1);
    test_rfind(sv, str3, 1, 1, 0);
    test_rfind(sv, str3, 1, 5, 0);
    test_rfind(sv, str3, 1, 9, SV::npos);
    test_rfind(sv, str3, 1, 10, SV::npos);
    test_rfind(sv, str4, 1, 0, 1);
    test_rfind(sv, str4, 1, 1, 0);
    test_rfind(sv, str4, 1, 10, SV::npos);
    test_rfind(sv, str4, 1, 19, SV::npos);
    test_rfind(sv, str4, 1, 20, SV::npos);
    test_rfind(sv, str1, 2, 0, 2);
    test_rfind(sv, str2, 2, 0, 2);
    test_rfind(sv, str2, 2, 1, 0);
    test_rfind(sv, str2, 2, 2, 0);
    test_rfind(sv, str2, 2, 4, 0);
    test_rfind(sv, str2, 2, 5, 0);
    test_rfind(sv, str3, 2, 0, 2);
    test_rfind(sv, str3, 2, 1, 0);
    test_rfind(sv, str3, 2, 5, 0);
    test_rfind(sv, str3, 2, 9, SV::npos);
    test_rfind(sv, str3, 2, 10, SV::npos);
    test_rfind(sv, str4, 2, 0, 2);
    test_rfind(sv, str4, 2, 1, 0);
    test_rfind(sv, str4, 2, 10, SV::npos);
    test_rfind(sv, str4, 2, 19, SV::npos);
    test_rfind(sv, str4, 2, 20, SV::npos);
    test_rfind(sv, str1, 4, 0, 4);
    test_rfind(sv, str2, 4, 0, 4);
    test_rfind(sv, str2, 4, 1, 0);
    test_rfind(sv, str2, 4, 2, 0);
    test_rfind(sv, str2, 4, 4, 0);
    test_rfind(sv, str2, 4, 5, 0);
    test_rfind(sv, str3, 4, 0, 4);
    test_rfind(sv, str3, 4, 1, 0);
    test_rfind(sv, str3, 4, 5, 0);
    test_rfind(sv, str3, 4, 9, SV::npos);
    test_rfind(sv, str3, 4, 10, SV::npos);
    test_rfind(sv, str4, 4, 0, 4);
    test_rfind(sv, str4, 4, 1, 0);
    test_rfind(sv, str4, 4, 10, SV::npos);
    test_rfind(sv, str4, 4, 19, SV::npos);
    test_rfind(sv, str4, 4, 20, SV::npos);
    test_rfind(sv, str1, 5, 0, 5);
    test_rfind(sv, str2, 5, 0, 5);
    test_rfind(sv, str2, 5, 1, 0);
    test_rfind(sv, str2, 5, 2, 0);
    test_rfind(sv, str2, 5, 4, 0);
    test_rfind(sv, str2, 5, 5, 0);
    test_rfind(sv, str3, 5, 0, 5);
    test_rfind(sv, str3, 5, 1, 0);
    test_rfind(sv, str3, 5, 5, 0);
    test_rfind(sv, str3, 5, 9, SV::npos);
    test_rfind(sv, str3, 5, 10, SV::npos);
    test_rfind(sv, str4, 5, 0, 5);
    test_rfind(sv, str4, 5, 1, 0);
    test_rfind(sv, str4, 5, 10, SV::npos);
    test_rfind(sv, str4, 5, 19, SV::npos);
    test_rfind(sv, str4, 5, 20, SV::npos);
    test_rfind(sv, str1, 6, 0, 5);
    test_rfind(sv, str2, 6, 0, 5);
    test_rfind(sv, str2, 6, 1, 0);
    test_rfind(sv, str2, 6, 2, 0);
    test_rfind(sv, str2, 6, 4, 0);
    test_rfind(sv, str2, 6, 5, 0);
    test_rfind(sv, str3, 6, 0, 5);
    test_rfind(sv, str3, 6, 1, 0);
    test_rfind(sv, str3, 6, 5, 0);
    test_rfind(sv, str3, 6, 9, SV::npos);
    test_rfind(sv, str3, 6, 10, SV::npos);
    test_rfind(sv, str4, 6, 0, 5);
    test_rfind(sv, str4, 6, 1, 0);
    test_rfind(sv, str4, 6, 10, SV::npos);
    test_rfind(sv, str4, 6, 19, SV::npos);
    test_rfind(sv, str4, 6, 20, SV::npos);
  }
  {
    SV sv{str3};
    test_rfind(sv, str1, 0, 0, 0);
    test_rfind(sv, str2, 0, 0, 0);
    test_rfind(sv, str2, 0, 1, 0);
    test_rfind(sv, str2, 0, 2, 0);
    test_rfind(sv, str2, 0, 4, 0);
    test_rfind(sv, str2, 0, 5, 0);
    test_rfind(sv, str3, 0, 0, 0);
    test_rfind(sv, str3, 0, 1, 0);
    test_rfind(sv, str3, 0, 5, 0);
    test_rfind(sv, str3, 0, 9, 0);
    test_rfind(sv, str3, 0, 10, 0);
    test_rfind(sv, str4, 0, 0, 0);
    test_rfind(sv, str4, 0, 1, 0);
    test_rfind(sv, str4, 0, 10, 0);
    test_rfind(sv, str4, 0, 19, SV::npos);
    test_rfind(sv, str4, 0, 20, SV::npos);
    test_rfind(sv, str1, 1, 0, 1);
    test_rfind(sv, str2, 1, 0, 1);
    test_rfind(sv, str2, 1, 1, 0);
    test_rfind(sv, str2, 1, 2, 0);
    test_rfind(sv, str2, 1, 4, 0);
    test_rfind(sv, str2, 1, 5, 0);
    test_rfind(sv, str3, 1, 0, 1);
    test_rfind(sv, str3, 1, 1, 0);
    test_rfind(sv, str3, 1, 5, 0);
    test_rfind(sv, str3, 1, 9, 0);
    test_rfind(sv, str3, 1, 10, 0);
    test_rfind(sv, str4, 1, 0, 1);
    test_rfind(sv, str4, 1, 1, 0);
    test_rfind(sv, str4, 1, 10, 0);
    test_rfind(sv, str4, 1, 19, SV::npos);
    test_rfind(sv, str4, 1, 20, SV::npos);
    test_rfind(sv, str1, 5, 0, 5);
    test_rfind(sv, str2, 5, 0, 5);
    test_rfind(sv, str2, 5, 1, 5);
    test_rfind(sv, str2, 5, 2, 5);
    test_rfind(sv, str2, 5, 4, 5);
    test_rfind(sv, str2, 5, 5, 5);
    test_rfind(sv, str3, 5, 0, 5);
    test_rfind(sv, str3, 5, 1, 5);
    test_rfind(sv, str3, 5, 5, 5);
    test_rfind(sv, str3, 5, 9, 0);
    test_rfind(sv, str3, 5, 10, 0);
    test_rfind(sv, str4, 5, 0, 5);
    test_rfind(sv, str4, 5, 1, 5);
    test_rfind(sv, str4, 5, 10, 0);
    test_rfind(sv, str4, 5, 19, SV::npos);
    test_rfind(sv, str4, 5, 20, SV::npos);
    test_rfind(sv, str1, 9, 0, 9);
    test_rfind(sv, str2, 9, 0, 9);
    test_rfind(sv, str2, 9, 1, 5);
    test_rfind(sv, str2, 9, 2, 5);
    test_rfind(sv, str2, 9, 4, 5);
    test_rfind(sv, str2, 9, 5, 5);
    test_rfind(sv, str3, 9, 0, 9);
    test_rfind(sv, str3, 9, 1, 5);
    test_rfind(sv, str3, 9, 5, 5);
    test_rfind(sv, str3, 9, 9, 0);
    test_rfind(sv, str3, 9, 10, 0);
    test_rfind(sv, str4, 9, 0, 9);
    test_rfind(sv, str4, 9, 1, 5);
    test_rfind(sv, str4, 9, 10, 0);
    test_rfind(sv, str4, 9, 19, SV::npos);
    test_rfind(sv, str4, 9, 20, SV::npos);
    test_rfind(sv, str1, 10, 0, 10);
    test_rfind(sv, str2, 10, 0, 10);
    test_rfind(sv, str2, 10, 1, 5);
    test_rfind(sv, str2, 10, 2, 5);
    test_rfind(sv, str2, 10, 4, 5);
    test_rfind(sv, str2, 10, 5, 5);
    test_rfind(sv, str3, 10, 0, 10);
    test_rfind(sv, str3, 10, 1, 5);
    test_rfind(sv, str3, 10, 5, 5);
    test_rfind(sv, str3, 10, 9, 0);
    test_rfind(sv, str3, 10, 10, 0);
    test_rfind(sv, str4, 10, 0, 10);
    test_rfind(sv, str4, 10, 1, 5);
    test_rfind(sv, str4, 10, 10, 0);
    test_rfind(sv, str4, 10, 19, SV::npos);
    test_rfind(sv, str4, 10, 20, SV::npos);
    test_rfind(sv, str1, 11, 0, 10);
    test_rfind(sv, str2, 11, 0, 10);
    test_rfind(sv, str2, 11, 1, 5);
    test_rfind(sv, str2, 11, 2, 5);
    test_rfind(sv, str2, 11, 4, 5);
    test_rfind(sv, str2, 11, 5, 5);
    test_rfind(sv, str3, 11, 0, 10);
    test_rfind(sv, str3, 11, 1, 5);
    test_rfind(sv, str3, 11, 5, 5);
    test_rfind(sv, str3, 11, 9, 0);
    test_rfind(sv, str3, 11, 10, 0);
    test_rfind(sv, str4, 11, 0, 10);
    test_rfind(sv, str4, 11, 1, 5);
    test_rfind(sv, str4, 11, 10, 0);
    test_rfind(sv, str4, 11, 19, SV::npos);
    test_rfind(sv, str4, 11, 20, SV::npos);
  }
  {
    SV sv{str4};
    test_rfind(sv, str1, 0, 0, 0);
    test_rfind(sv, str2, 0, 0, 0);
    test_rfind(sv, str2, 0, 1, 0);
    test_rfind(sv, str2, 0, 2, 0);
    test_rfind(sv, str2, 0, 4, 0);
    test_rfind(sv, str2, 0, 5, 0);
    test_rfind(sv, str3, 0, 0, 0);
    test_rfind(sv, str3, 0, 1, 0);
    test_rfind(sv, str3, 0, 5, 0);
    test_rfind(sv, str3, 0, 9, 0);
    test_rfind(sv, str3, 0, 10, 0);
    test_rfind(sv, str4, 0, 0, 0);
    test_rfind(sv, str4, 0, 1, 0);
    test_rfind(sv, str4, 0, 10, 0);
    test_rfind(sv, str4, 0, 19, 0);
    test_rfind(sv, str4, 0, 20, 0);
    test_rfind(sv, str1, 1, 0, 1);
    test_rfind(sv, str2, 1, 0, 1);
    test_rfind(sv, str2, 1, 1, 0);
    test_rfind(sv, str2, 1, 2, 0);
    test_rfind(sv, str2, 1, 4, 0);
    test_rfind(sv, str2, 1, 5, 0);
    test_rfind(sv, str3, 1, 0, 1);
    test_rfind(sv, str3, 1, 1, 0);
    test_rfind(sv, str3, 1, 5, 0);
    test_rfind(sv, str3, 1, 9, 0);
    test_rfind(sv, str3, 1, 10, 0);
    test_rfind(sv, str4, 1, 0, 1);
    test_rfind(sv, str4, 1, 1, 0);
    test_rfind(sv, str4, 1, 10, 0);
    test_rfind(sv, str4, 1, 19, 0);
    test_rfind(sv, str4, 1, 20, 0);
    test_rfind(sv, str1, 10, 0, 10);
    test_rfind(sv, str2, 10, 0, 10);
    test_rfind(sv, str2, 10, 1, 10);
    test_rfind(sv, str2, 10, 2, 10);
    test_rfind(sv, str2, 10, 4, 10);
    test_rfind(sv, str2, 10, 5, 10);
    test_rfind(sv, str3, 10, 0, 10);
    test_rfind(sv, str3, 10, 1, 10);
    test_rfind(sv, str3, 10, 5, 10);
    test_rfind(sv, str3, 10, 9, 10);
    test_rfind(sv, str3, 10, 10, 10);
    test_rfind(sv, str4, 10, 0, 10);
    test_rfind(sv, str4, 10, 1, 10);
    test_rfind(sv, str4, 10, 10, 10);
    test_rfind(sv, str4, 10, 19, 0);
    test_rfind(sv, str4, 10, 20, 0);
    test_rfind(sv, str1, 19, 0, 19);
    test_rfind(sv, str2, 19, 0, 19);
    test_rfind(sv, str2, 19, 1, 15);
    test_rfind(sv, str2, 19, 2, 15);
    test_rfind(sv, str2, 19, 4, 15);
    test_rfind(sv, str2, 19, 5, 15);
    test_rfind(sv, str3, 19, 0, 19);
    test_rfind(sv, str3, 19, 1, 15);
    test_rfind(sv, str3, 19, 5, 15);
    test_rfind(sv, str3, 19, 9, 10);
    test_rfind(sv, str3, 19, 10, 10);
    test_rfind(sv, str4, 19, 0, 19);
    test_rfind(sv, str4, 19, 1, 15);
    test_rfind(sv, str4, 19, 10, 10);
    test_rfind(sv, str4, 19, 19, 0);
    test_rfind(sv, str4, 19, 20, 0);
    test_rfind(sv, str1, 20, 0, 20);
    test_rfind(sv, str2, 20, 0, 20);
    test_rfind(sv, str2, 20, 1, 15);
    test_rfind(sv, str2, 20, 2, 15);
    test_rfind(sv, str2, 20, 4, 15);
    test_rfind(sv, str2, 20, 5, 15);
    test_rfind(sv, str3, 20, 0, 20);
    test_rfind(sv, str3, 20, 1, 15);
    test_rfind(sv, str3, 20, 5, 15);
    test_rfind(sv, str3, 20, 9, 10);
    test_rfind(sv, str3, 20, 10, 10);
    test_rfind(sv, str4, 20, 0, 20);
    test_rfind(sv, str4, 20, 1, 15);
    test_rfind(sv, str4, 20, 10, 10);
    test_rfind(sv, str4, 20, 19, 0);
    test_rfind(sv, str4, 20, 20, 0);
    test_rfind(sv, str1, 21, 0, 20);
    test_rfind(sv, str2, 21, 0, 20);
    test_rfind(sv, str2, 21, 1, 15);
    test_rfind(sv, str2, 21, 2, 15);
    test_rfind(sv, str2, 21, 4, 15);
    test_rfind(sv, str2, 21, 5, 15);
    test_rfind(sv, str3, 21, 0, 20);
    test_rfind(sv, str3, 21, 1, 15);
    test_rfind(sv, str3, 21, 5, 15);
    test_rfind(sv, str3, 21, 9, 10);
    test_rfind(sv, str3, 21, 10, 10);
    test_rfind(sv, str4, 21, 0, 20);
    test_rfind(sv, str4, 21, 1, 15);
    test_rfind(sv, str4, 21, 10, 10);
    test_rfind(sv, str4, 21, 19, 0);
    test_rfind(sv, str4, 21, 20, 0);
  }
}

__host__ __device__ constexpr bool test()
{
  test_rfind<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_rfind<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_rfind<cuda::std::u16string_view>();
  test_rfind<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_rfind<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

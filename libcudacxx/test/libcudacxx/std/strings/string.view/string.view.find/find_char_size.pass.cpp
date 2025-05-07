//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type find(charT c, size_type pos = 0) const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_find(const SV& sv, typename SV::value_type c, typename SV::size_type x)
{
  assert(sv.find(c) == x);
  if (x != SV::npos)
  {
    assert(x + 1 <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void
test_find(const SV& sv, typename SV::value_type c, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.find(c, pos) == x);
  if (x != SV::npos)
  {
    assert(pos <= x && x + 1 <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_find()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find(CharT{}))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find(CharT{}, SizeT{}))>);

  static_assert(noexcept(SV{}.find(CharT{})));
  static_assert(noexcept(SV{}.find(CharT{}, SizeT{})));

  const CharT c = TEST_CHARLIT(CharT, 'c');

  {
    SV sv{TEST_STRLIT(CharT, "")};
    test_find(sv, c, SV::npos);
    test_find(sv, c, 0, SV::npos);
    test_find(sv, c, 1, SV::npos);
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcde")};
    test_find(sv, c, 2);
    test_find(sv, c, 0, 2);
    test_find(sv, c, 1, 2);
    test_find(sv, c, 2, 2);
    test_find(sv, c, 4, SV::npos);
    test_find(sv, c, 5, SV::npos);
    test_find(sv, c, 6, SV::npos);
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcdeabcde")};
    test_find(sv, c, 2);
    test_find(sv, c, 0, 2);
    test_find(sv, c, 1, 2);
    test_find(sv, c, 5, 7);
    test_find(sv, c, 9, SV::npos);
    test_find(sv, c, 10, SV::npos);
    test_find(sv, c, 11, SV::npos);
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcdeabcdeabcdeabcde")};
    test_find(sv, c, 2);
    test_find(sv, c, 0, 2);
    test_find(sv, c, 1, 2);
    test_find(sv, c, 10, 12);
    test_find(sv, c, 19, SV::npos);
    test_find(sv, c, 20, SV::npos);
    test_find(sv, c, 21, SV::npos);
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

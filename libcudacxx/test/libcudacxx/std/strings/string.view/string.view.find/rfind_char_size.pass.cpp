//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type rfind(charT c, size_type pos = npos) const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_rfind(const SV& sv, typename SV::value_type c, typename SV::size_type x)
{
  assert(sv.rfind(c) == x);
  if (x != SV::npos)
  {
    assert(x + 1 <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void
test_rfind(const SV& sv, typename SV::value_type c, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.rfind(c, pos) == x);
  if (x != SV::npos)
  {
    assert(x <= pos && x + 1 <= sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_rfind()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.rfind(CharT{}))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.rfind(CharT{}, SizeT{}))>);

  static_assert(noexcept(SV{}.rfind(CharT{})));
  static_assert(noexcept(SV{}.rfind(CharT{}, SizeT{})));

  const CharT c = TEST_CHARLIT(CharT, 'b');

  {
    SV sv{TEST_STRLIT(CharT, "")};
    test_rfind(sv, c, SV::npos);
    test_rfind(sv, c, 0, SV::npos);
    test_rfind(sv, c, 1, SV::npos);
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcde")};
    test_rfind(sv, c, 1);
    test_rfind(sv, c, 0, SV::npos);
    test_rfind(sv, c, 1, 1);
    test_rfind(sv, c, 2, 1);
    test_rfind(sv, c, 4, 1);
    test_rfind(sv, c, 5, 1);
    test_rfind(sv, c, 6, 1);
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcdeabcde")};
    test_rfind(sv, c, 6);
    test_rfind(sv, c, 0, SV::npos);
    test_rfind(sv, c, 1, 1);
    test_rfind(sv, c, 5, 1);
    test_rfind(sv, c, 9, 6);
    test_rfind(sv, c, 10, 6);
    test_rfind(sv, c, 11, 6);
  }
  {
    SV sv{TEST_STRLIT(CharT, "abcdeabcdeabcdeabcde")};
    test_rfind(sv, c, 16);
    test_rfind(sv, c, 0, SV::npos);
    test_rfind(sv, c, 1, 1);
    test_rfind(sv, c, 10, 6);
    test_rfind(sv, c, 19, 16);
    test_rfind(sv, c, 20, 16);
    test_rfind(sv, c, 21, 16);
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

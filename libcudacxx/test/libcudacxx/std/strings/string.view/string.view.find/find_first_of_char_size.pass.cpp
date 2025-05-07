//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type find_first_of(charT c, size_type pos = 0) const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_find_first_of(const SV& sv, typename SV::value_type c, typename SV::size_type x)
{
  assert(sv.find_first_of(c) == x);
  if (x != SV::npos)
  {
    assert(x < sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void
test_find_first_of(const SV& sv, typename SV::value_type c, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.find_first_of(c, pos) == x);
  if (x != SV::npos)
  {
    assert(pos <= x && x < sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_find_first_of()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_first_of(CharT{}))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_first_of(CharT{}, SizeT{}))>);

  static_assert(noexcept(SV{}.find_first_of(CharT{})));
  static_assert(noexcept(SV{}.find_first_of(CharT{}, SizeT{})));

  const CharT c = TEST_CHARLIT(CharT, 'e');

  test_find_first_of(SV(TEST_STRLIT(CharT, "")), c, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "")), c, 0, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "")), c, 1, SV::npos);

  test_find_first_of(SV(TEST_STRLIT(CharT, "csope")), c, 4);
  test_find_first_of(SV(TEST_STRLIT(CharT, "kitcj")), c, 0, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "qkamf")), c, 1, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "nhmko")), c, 2, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "tpsaf")), c, 4, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "lahfb")), c, 5, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "irkhs")), c, 6, SV::npos);

  test_find_first_of(SV(TEST_STRLIT(CharT, "gfsmthlkon")), c, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "gmfhdaipsr")), c, 0, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "kantesmpgj")), c, 1, 4);
  test_find_first_of(SV(TEST_STRLIT(CharT, "odaftiegpm")), c, 5, 6);
  test_find_first_of(SV(TEST_STRLIT(CharT, "oknlrstdpi")), c, 9, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "eolhfgpjqk")), c, 10, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "pcdrofikas")), c, 11, SV::npos);

  test_find_first_of(SV(TEST_STRLIT(CharT, "laenfsbridchgotmkqpj")), c, 2);
  test_find_first_of(SV(TEST_STRLIT(CharT, "nbatdlmekrgcfqsophij")), c, 0, 7);
  test_find_first_of(SV(TEST_STRLIT(CharT, "bnrpehidofmqtcksjgla")), c, 1, 4);
  test_find_first_of(SV(TEST_STRLIT(CharT, "jdmciepkaqgotsrfnhlb")), c, 10, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "jtdaefblsokrmhpgcnqi")), c, 19, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "hkbgspofltajcnedqmri")), c, 20, SV::npos);
  test_find_first_of(SV(TEST_STRLIT(CharT, "oselktgbcapndfjihrmq")), c, 21, SV::npos);
}

__host__ __device__ constexpr bool test()
{
  test_find_first_of<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_find_first_of<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_find_first_of<cuda::std::u16string_view>();
  test_find_first_of<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_find_first_of<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

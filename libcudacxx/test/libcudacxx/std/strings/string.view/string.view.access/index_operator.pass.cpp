//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr const_reference operator[](size_type pos) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_index_operator()
{
  using CharT    = typename SV::value_type;
  using SizeT    = typename SV::size_type;
  using ConstRef = typename SV::const_reference;

  static_assert(cuda::std::is_same_v<ConstRef, decltype(SV{}.operator[](SizeT{}))>);
  static_assert(noexcept(SV{}.operator[](SizeT{})));

  const CharT* str = TEST_STRLIT(CharT, "Hello world!");

  SV sv{str};
  assert(sv[0] == str[0]);
  assert(sv[1] == str[1]);
  assert(sv[4] == str[4]);
  assert(sv[8] == str[8]);
  assert(sv[11] == str[11]);
}

__host__ __device__ constexpr bool test()
{
  test_index_operator<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_index_operator<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_index_operator<cuda::std::u16string_view>();
  test_index_operator<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_index_operator<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

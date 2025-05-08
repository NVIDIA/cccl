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

// constexpr bool empty() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_empty()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<bool, decltype(SV{}.empty())>);
  static_assert(noexcept(SV{}.empty()));

  {
    SV sv;
    assert(sv.empty());
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "Hello world!");

    SV sv{str};
    assert(!sv.empty());
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "Hello world!");

    SV sv{str, SizeT{0}};
    assert(sv.empty());
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "Hello world!");

    SV sv{str, SizeT{1}};
    assert(!sv.empty());
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "");

    SV sv{str};
    assert(sv.empty());
  }
}

__host__ __device__ constexpr bool test()
{
  test_empty<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_empty<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_empty<cuda::std::u16string_view>();
  test_empty<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_empty<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

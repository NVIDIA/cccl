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

// constexpr const_pointer front() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_data()
{
  using CharT = typename SV::value_type;

  static_assert(cuda::std::is_same_v<const CharT*, decltype(SV{}.data())>);
  static_assert(noexcept(SV{}.data()));

  {
    const CharT* null_str = nullptr;
    SV sv{null_str, 0};
    assert(sv.data() == nullptr);
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "a");
    SV sv{str};
    assert(sv.data() == str);
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "Hello world!");
    SV sv{str};
    assert(sv.data() == str);
  }
}

__host__ __device__ constexpr bool test()
{
  test_data<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_data<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_data<cuda::std::u16string_view>();
  test_data<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_data<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

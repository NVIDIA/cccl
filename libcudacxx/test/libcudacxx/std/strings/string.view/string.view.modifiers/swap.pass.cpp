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

// constexpr void swap(basic_string_view& s) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_swap()
{
  using CharT  = typename SV::value_type;
  using Traits = typename SV::traits_type;

  static_assert(cuda::std::is_same_v<void, decltype(SV{}.swap(cuda::std::declval<SV&>()))>);
  static_assert(noexcept(SV{}.swap(cuda::std::declval<SV&>())));

  const CharT* str1 = TEST_STRLIT(CharT, "Hello");
  const CharT* str2 = TEST_STRLIT(CharT, "World!");

  const auto str1_size = Traits::length(str1);
  const auto str2_size = Traits::length(str2);

  {
    SV sv1{str1};
    SV sv2{str2};

    assert(sv1.data() == str1);
    assert(sv1.size() == str1_size);
    assert(sv2.data() == str2);
    assert(sv2.size() == str2_size);

    sv1.swap(sv2);

    assert(sv1.data() == str2);
    assert(sv1.size() == str2_size);
    assert(sv2.data() == str1);
    assert(sv2.size() == str1_size);
  }
  {
    SV sv1;
    SV sv2{str2};

    assert(sv1.data() == nullptr);
    assert(sv1.size() == 0);
    assert(sv2.data() == str2);
    assert(sv2.size() == str2_size);

    sv1.swap(sv2);

    assert(sv1.data() == str2);
    assert(sv1.size() == str2_size);
    assert(sv2.data() == nullptr);
    assert(sv2.size() == 0);
  }
  {
    SV sv1{str1};
    SV sv2;

    assert(sv1.data() == str1);
    assert(sv1.size() == str1_size);
    assert(sv2.data() == nullptr);
    assert(sv2.size() == 0);

    sv1.swap(sv2);

    assert(sv1.data() == nullptr);
    assert(sv1.size() == 0);
    assert(sv2.data() == str1);
    assert(sv2.size() == str1_size);
  }
  {
    SV sv1;
    SV sv2;

    assert(sv1.data() == nullptr);
    assert(sv1.size() == 0);
    assert(sv2.data() == nullptr);
    assert(sv2.size() == 0);

    sv1.swap(sv2);

    assert(sv1.data() == nullptr);
    assert(sv1.size() == 0);
    assert(sv2.data() == nullptr);
    assert(sv2.size() == 0);
  }
}

__host__ __device__ constexpr bool test()
{
  test_swap<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_swap<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_swap<cuda::std::u16string_view>();
  test_swap<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_swap<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

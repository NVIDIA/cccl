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

// constexpr basic_string_view(const CharT* str, size_type len);

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_sized_str_constructor()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_constructible_v<SV, const CharT*, SizeT>);
  static_assert(noexcept(SV{cuda::std::declval<const CharT*>(), cuda::std::declval<SizeT>()}));

  const CharT* str = TEST_STRLIT(CharT, "Hello world!");
  {
    SV sv{str, SizeT{0}};
    assert(sv.data() == str);
    assert(sv.size() == SizeT{0});
  }
  {
    SV sv{str, SizeT{1}};
    assert(sv.data() == str);
    assert(sv.size() == SizeT{1});
  }
  {
    SV sv{str, SizeT{11}};
    assert(sv.data() == str);
    assert(sv.size() == SizeT{11});
  }

  const CharT* null_str = nullptr;
  {
    SV sv{null_str, SizeT{0}};
    assert(sv.data() == nullptr);
    assert(sv.size() == SizeT{0});
  }
}

__host__ __device__ constexpr bool test()
{
  test_sized_str_constructor<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_sized_str_constructor<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_sized_str_constructor<cuda::std::u16string_view>();
  test_sized_str_constructor<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_sized_str_constructor<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

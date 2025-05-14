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

// constexpr basic_string_view& operator=(const basic_string_view&) noexcept = default;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_assignment()
{
  using CharT = typename SV::value_type;

  static_assert(cuda::std::is_trivially_copy_assignable_v<SV>);
  static_assert(noexcept(SV{}.operator=(SV{})));

  const CharT* str = TEST_STRLIT(CharT, "Hello world!");

  SV sv{str};

  SV sv_copy;
  sv_copy = sv;

  assert(sv_copy.data() == sv.data());
  assert(sv_copy.size() == sv.size());
}

__host__ __device__ constexpr bool test()
{
  test_assignment<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_assignment<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_assignment<cuda::std::u16string_view>();
  test_assignment<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_assignment<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

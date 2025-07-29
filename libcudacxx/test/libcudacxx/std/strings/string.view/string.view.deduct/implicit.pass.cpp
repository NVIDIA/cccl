//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// basic_string_view

// Make sure that the implicitly-generated CTAD works.

#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class CharT>
__host__ __device__ constexpr void test_implicit_ctad()
{
  const CharT* str = TEST_STRLIT(CharT, "Hello world!");
  cuda::std::basic_string_view sv{str};
  static_assert(cuda::std::is_same_v<decltype(sv), cuda::std::basic_string_view<CharT>>);
}

__host__ __device__ constexpr bool test()
{
  test_implicit_ctad<char>();
#if _CCCL_HAS_CHAR8_T()
  test_implicit_ctad<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_implicit_ctad<char16_t>();
  test_implicit_ctad<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_implicit_ctad<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

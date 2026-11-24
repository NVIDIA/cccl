//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string/literal.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class CharT>
TEST_FUNC constexpr void test()
{
  decltype(auto) str = _CCCL_STRLIT(CharT, "hello");
  static_assert(cuda::std::is_same_v<const CharT(&)[6], decltype(str)>);

  assert(str[0] == CharT{'h'});
  assert(str[1] == CharT{'e'});
  assert(str[2] == CharT{'l'});
  assert(str[3] == CharT{'l'});
  assert(str[4] == CharT{'o'});
  assert(str[5] == CharT{'\0'});
}

TEST_FUNC constexpr bool test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

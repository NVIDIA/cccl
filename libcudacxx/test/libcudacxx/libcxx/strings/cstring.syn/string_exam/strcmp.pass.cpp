//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string/constexpr_c_functions.h>
#include <cuda/std/cassert>

template <class T>
__host__ __device__ constexpr void test_strcmp(const T* lhs, const T* rhs, int expected)
{
  const auto ret = cuda::std::__cccl_strcmp(lhs, rhs);

  if (expected == 0)
  {
    assert(ret == 0);
  }
  else if (expected < 0)
  {
    assert(ret < 0);
  }
  else
  {
    assert(ret > 0);
  }
}

__host__ __device__ constexpr bool test()
{
  // char
  test_strcmp<char>("", "", 0);
  test_strcmp<char>("a", "", 1);
  test_strcmp<char>("", "a", -1);
  test_strcmp<char>("hi", "hi", 0);
  test_strcmp<char>("hi", "ho", -1);
  test_strcmp<char>("ho", "hi", 1);
  test_strcmp<char>("abcde", "abcde", 0);
  test_strcmp<char>("abcd1", "abcd0", 1);
  test_strcmp<char>("abcd0", "abcd1", -1);
  test_strcmp<char>("ab1de", "abcd0", -1);

#if _LIBCUDACXX_HAS_CHAR8_T()
  // char8_t
  test_strcmp<char8_t>(u8"", u8"", 0);
  test_strcmp<char8_t>(u8"a", u8"", 1);
  test_strcmp<char8_t>(u8"", u8"a", -1);
  test_strcmp<char8_t>(u8"hi", u8"hi", 0);
  test_strcmp<char8_t>(u8"hi", u8"ho", -1);
  test_strcmp<char8_t>(u8"ho", u8"hi", 1);
  test_strcmp<char8_t>(u8"abcde", u8"abcde", 0);
  test_strcmp<char8_t>(u8"abcd1", u8"abcd0", 1);
  test_strcmp<char8_t>(u8"abcd0", u8"abcd1", -1);
  test_strcmp<char8_t>(u8"ab1de", u8"abcd0", -1);
#endif // _LIBCUDACXX_HAS_CHAR8_T()

  // char16_t
  test_strcmp<char16_t>(u"", u"", 0);
  test_strcmp<char16_t>(u"a", u"", 1);
  test_strcmp<char16_t>(u"", u"a", -1);
  test_strcmp<char16_t>(u"hi", u"hi", 0);
  test_strcmp<char16_t>(u"hi", u"ho", -1);
  test_strcmp<char16_t>(u"ho", u"hi", 1);
  test_strcmp<char16_t>(u"abcde", u"abcde", 0);
  test_strcmp<char16_t>(u"abcd1", u"abcd0", 1);
  test_strcmp<char16_t>(u"abcd0", u"abcd1", -1);
  test_strcmp<char16_t>(u"ab1de", u"abcd0", -1);

  // char32_t
  test_strcmp<char32_t>(U"", U"", 0);
  test_strcmp<char32_t>(U"a", U"", 1);
  test_strcmp<char32_t>(U"", U"a", -1);
  test_strcmp<char32_t>(U"hi", U"hi", 0);
  test_strcmp<char32_t>(U"hi", U"ho", -1);
  test_strcmp<char32_t>(U"ho", U"hi", 1);
  test_strcmp<char32_t>(U"abcde", U"abcde", 0);
  test_strcmp<char32_t>(U"abcd1", U"abcd0", 1);
  test_strcmp<char32_t>(U"abcd0", U"abcd1", -1);
  test_strcmp<char32_t>(U"ab1de", U"abcd0", -1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

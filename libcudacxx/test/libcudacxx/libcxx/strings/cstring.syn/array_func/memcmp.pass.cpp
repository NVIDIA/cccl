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
#include <cuda/std/cstddef>

template <class T>
__host__ __device__ constexpr void test_memcmp(const T* lhs, const T* rhs, size_t n, int expected)
{
  const auto ret = cuda::std::__cccl_memcmp(lhs, rhs, n);

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
  test_memcmp<char>("abcde", "abcde", 5, 0);
  test_memcmp<char>("abcd1", "abcd0", 5, 1);
  test_memcmp<char>("abcd0", "abcd1", 5, -1);
  test_memcmp<char>("abcd1", "abcd0", 4, 0);
  test_memcmp<char>("abcd0", "abcd1", 4, 0);
  test_memcmp<char>("abcde", "fghij", 5, -1);
  test_memcmp<char>("abcde", "fghij", 0, 0);
  test_memcmp<char>(nullptr, nullptr, 0, 0);

#if _LIBCUDACXX_HAS_CHAR8_T()
  // char8_t
  test_memcmp<char8_t>(u8"abcde", u8"abcde", 5, 0);
  test_memcmp<char8_t>(u8"abcd1", u8"abcd0", 5, 1);
  test_memcmp<char8_t>(u8"abcd0", u8"abcd1", 5, -1);
  test_memcmp<char8_t>(u8"abcd1", u8"abcd0", 4, 0);
  test_memcmp<char8_t>(u8"abcd0", u8"abcd1", 4, 0);
  test_memcmp<char8_t>(u8"abcde", u8"fghij", 5, -1);
  test_memcmp<char8_t>(u8"abcde", u8"fghij", 0, 0);
  test_memcmp<char8_t>(nullptr, nullptr, 0, 0);
#endif // _LIBCUDACXX_HAS_CHAR8_T()

  // char16_t
  test_memcmp<char16_t>(u"abcde", u"abcde", 5, 0);
  test_memcmp<char16_t>(u"abcd1", u"abcd0", 5, 1);
  test_memcmp<char16_t>(u"abcd0", u"abcd1", 5, -1);
  test_memcmp<char16_t>(u"abcd1", u"abcd0", 4, 0);
  test_memcmp<char16_t>(u"abcd0", u"abcd1", 4, 0);
  test_memcmp<char16_t>(u"abcde", u"fghij", 5, -1);
  test_memcmp<char16_t>(u"abcde", u"fghij", 0, 0);
  test_memcmp<char16_t>(nullptr, nullptr, 0, 0);

  // char32_t
  test_memcmp<char32_t>(U"abcde", U"abcde", 5, 0);
  test_memcmp<char32_t>(U"abcd1", U"abcd0", 5, 1);
  test_memcmp<char32_t>(U"abcd0", U"abcd1", 5, -1);
  test_memcmp<char32_t>(U"abcd1", U"abcd0", 4, 0);
  test_memcmp<char32_t>(U"abcd0", U"abcd1", 4, 0);
  test_memcmp<char32_t>(U"abcde", U"fghij", 5, -1);
  test_memcmp<char32_t>(U"abcde", U"fghij", 0, 0);
  test_memcmp<char32_t>(nullptr, nullptr, 0, 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

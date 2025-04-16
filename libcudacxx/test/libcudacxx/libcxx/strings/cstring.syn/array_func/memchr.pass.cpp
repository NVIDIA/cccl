//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string/constexpr_c_functions.h>
#include <cuda/std/type_traits>

constexpr int not_found = -1;

template <class T>
__host__ __device__ constexpr void test_memchr(const T* ptr, T c, size_t n, int expected_pos)
{
  const T* ret = cuda::std::__cccl_memchr<const T>(ptr, c, n);

  if (expected_pos == not_found)
  {
    assert(ret == nullptr);
  }
  else
  {
    assert(ret != nullptr);
    assert(ret == ptr + expected_pos);
  }
}

__host__ __device__ constexpr bool test()
{
  // char
  test_memchr<char>("abcde", '\0', 6, 5);
  test_memchr<char>("abcde", '\0', 5, not_found);
  test_memchr<char>("aaabb", 'b', 5, 3);
  test_memchr<char>("aaabb", 'b', 4, 3);
  test_memchr<char>("aaabb", 'b', 3, not_found);
  test_memchr<char>("aaaa", 'b', 4, not_found);
  test_memchr<char>("aaaa", 'a', 0, not_found);

#if _LIBCUDACXX_HAS_CHAR8_T()
  // char8_t
  test_memchr<char8_t>(u8"abcde", u8'\0', 6, 5);
  test_memchr<char8_t>(u8"abcde", u8'\0', 5, not_found);
  test_memchr<char8_t>(u8"aaabb", u8'b', 5, 3);
  test_memchr<char8_t>(u8"aaabb", u8'b', 4, 3);
  test_memchr<char8_t>(u8"aaabb", u8'b', 3, not_found);
  test_memchr<char8_t>(u8"aaaa", u8'b', 4, not_found);
  test_memchr<char8_t>(u8"aaaa", u8'a', 0, not_found);
#endif // _LIBCUDACXX_HAS_CHAR8_T()

  // char16_t
  test_memchr<char16_t>(u"abcde", u'\0', 6, 5);
  test_memchr<char16_t>(u"abcde", u'\0', 5, not_found);
  test_memchr<char16_t>(u"aaabb", u'b', 5, 3);
  test_memchr<char16_t>(u"aaabb", u'b', 4, 3);
  test_memchr<char16_t>(u"aaabb", u'b', 3, not_found);
  test_memchr<char16_t>(u"aaaa", u'b', 4, not_found);
  test_memchr<char16_t>(u"aaaa", u'a', 0, not_found);

  // char32_t
  test_memchr<char32_t>(U"abcde", U'\0', 6, 5);
  test_memchr<char32_t>(U"abcde", U'\0', 5, not_found);
  test_memchr<char32_t>(U"aaabb", U'b', 5, 3);
  test_memchr<char32_t>(U"aaabb", U'b', 4, 3);
  test_memchr<char32_t>(U"aaabb", U'b', 3, not_found);
  test_memchr<char32_t>(U"aaaa", U'b', 4, not_found);
  test_memchr<char32_t>(U"aaaa", U'a', 0, not_found);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

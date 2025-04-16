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
__host__ __device__ constexpr void test_strlen(const T* str, cuda::std::size_t expected)
{
  const auto ret = cuda::std::__cccl_strlen(str);
  assert(ret == expected);
}

__host__ __device__ constexpr bool test()
{
  // char
  test_strlen<char>("", 0);
  test_strlen<char>("a", 1);
  test_strlen<char>("hi", 2);
  test_strlen<char>("asdfgh", 6);
  test_strlen<char>("asdfasdfasdfgweyr", 17);
  test_strlen<char>("asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char>("\0\0", 0);
  test_strlen<char>("a\0", 1);
  test_strlen<char>("h\0i", 1);
  test_strlen<char>("asdf\0g\0h", 4);
  test_strlen<char>("asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char>("asdfasdfasdfgweyr1239859102384\0\0", 30);

#if _LIBCUDACXX_HAS_CHAR8_T()
  // char8_t
  test_strlen<char8_t>(u8"", 0);
  test_strlen<char8_t>(u8"a", 1);
  test_strlen<char8_t>(u8"hi", 2);
  test_strlen<char8_t>(u8"asdfgh", 6);
  test_strlen<char8_t>(u8"asdfasdfasdfgweyr", 17);
  test_strlen<char8_t>(u8"asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char8_t>(u8"\0\0", 0);
  test_strlen<char8_t>(u8"a\0", 1);
  test_strlen<char8_t>(u8"h\0i", 1);
  test_strlen<char8_t>(u8"asdf\0g\0h", 4);
  test_strlen<char8_t>(u8"asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char8_t>(u8"asdfasdfasdfgweyr1239859102384\0\0", 30);
#endif // _LIBCUDACXX_HAS_CHAR8_T()

  // char16_t
  test_strlen<char16_t>(u"", 0);
  test_strlen<char16_t>(u"a", 1);
  test_strlen<char16_t>(u"hi", 2);
  test_strlen<char16_t>(u"asdfgh", 6);
  test_strlen<char16_t>(u"asdfasdfasdfgweyr", 17);
  test_strlen<char16_t>(u"asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char16_t>(u"\0\0", 0);
  test_strlen<char16_t>(u"a\0", 1);
  test_strlen<char16_t>(u"h\0i", 1);
  test_strlen<char16_t>(u"asdf\0g\0h", 4);
  test_strlen<char16_t>(u"asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char16_t>(u"asdfasdfasdfgweyr1239859102384\0\0", 30);

  // char32_t
  test_strlen<char32_t>(U"", 0);
  test_strlen<char32_t>(U"a", 1);
  test_strlen<char32_t>(U"hi", 2);
  test_strlen<char32_t>(U"asdfgh", 6);
  test_strlen<char32_t>(U"asdfasdfasdfgweyr", 17);
  test_strlen<char32_t>(U"asdfasdfasdfgweyr1239859102384", 30);
  test_strlen<char32_t>(U"\0\0", 0);
  test_strlen<char32_t>(U"a\0", 1);
  test_strlen<char32_t>(U"h\0i", 1);
  test_strlen<char32_t>(U"asdf\0g\0h", 4);
  test_strlen<char32_t>(U"asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen<char32_t>(U"asdfasdfasdfgweyr1239859102384\0\0", 30);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

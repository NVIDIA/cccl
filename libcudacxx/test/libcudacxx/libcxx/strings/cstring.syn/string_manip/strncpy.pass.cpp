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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T, cuda::std::size_t... N>
__host__ __device__ constexpr bool equal_buffers(const T* lhs, const T* rhs, cuda::std::index_sequence<N...>)
{
  return ((lhs[N] == rhs[N]) && ...);
}

template <class T, cuda::std::size_t N>
__host__ __device__ constexpr void test_strncpy(const T* str, cuda::std::size_t count, const T (&ref)[N])
{
  T buff[N]{};
  for (cuda::std::size_t i = 0; i < N; ++i)
  {
    buff[i] = T('x');
  }

  const auto ret = cuda::std::__cccl_constexpr_strncpy(buff, str, count);
  assert(ret == buff);
  assert(equal_buffers(buff, ref, cuda::std::make_index_sequence<N - 1>{}));
}

__host__ __device__ constexpr bool test()
{
  // char8
  test_strncpy<char>("", 0, "xxx");
  test_strncpy<char>("", 1, "\0xx");
  test_strncpy<char>("", 2, "\0\0x");
  test_strncpy<char>("", 3, "\0\0\0");
  test_strncpy<char>("a", 0, "xxx");
  test_strncpy<char>("a", 1, "axx");
  test_strncpy<char>("a", 2, "a\0x");
  test_strncpy<char>("a", 3, "a\0\0");
  test_strncpy<char>("\0a", 0, "xxx");
  test_strncpy<char>("\0a", 1, "\0xx");
  test_strncpy<char>("\0a", 2, "\0\0x");
  test_strncpy<char>("\0a", 3, "\0\0\0");
  test_strncpy<char>("hello", 5, "helloxxx");
  test_strncpy<char>("hello", 6, "hello\0xx");
  test_strncpy<char>("hello", 7, "hello\0\0x");
  test_strncpy<char>("hello", 8, "hello\0\0\0");
  test_strncpy<char>("hell\0o", 4, "hellxxxx");
  test_strncpy<char>("hell\0o", 5, "hell\0xx");
  test_strncpy<char>("hell\0o", 6, "hell\0\0x");
  test_strncpy<char>("hell\0o", 7, "hell\0\0\0");

  // char16
  test_strncpy<char16_t>(u"", 0, u"xxx");
  test_strncpy<char16_t>(u"", 1, u"\0xx");
  test_strncpy<char16_t>(u"", 2, u"\0\0x");
  test_strncpy<char16_t>(u"", 3, u"\0\0\0");
  test_strncpy<char16_t>(u"a", 0, u"xxx");
  test_strncpy<char16_t>(u"a", 1, u"axx");
  test_strncpy<char16_t>(u"a", 2, u"a\0x");
  test_strncpy<char16_t>(u"a", 3, u"a\0\0");
  test_strncpy<char16_t>(u"\0a", 0, u"xxx");
  test_strncpy<char16_t>(u"\0a", 1, u"\0xx");
  test_strncpy<char16_t>(u"\0a", 2, u"\0\0x");
  test_strncpy<char16_t>(u"\0a", 3, u"\0\0\0");
  test_strncpy<char16_t>(u"hello", 5, u"helloxxx");
  test_strncpy<char16_t>(u"hello", 6, u"hello\0xx");
  test_strncpy<char16_t>(u"hello", 7, u"hello\0\0x");
  test_strncpy<char16_t>(u"hello", 8, u"hello\0\0\0");
  test_strncpy<char16_t>(u"hell\0o", 4, u"hellxxxx");
  test_strncpy<char16_t>(u"hell\0o", 5, u"hell\0xx");
  test_strncpy<char16_t>(u"hell\0o", 6, u"hell\0\0x");
  test_strncpy<char16_t>(u"hell\0o", 7, u"hell\0\0\0");

  // char32
  test_strncpy<char32_t>(U"", 0, U"xxx");
  test_strncpy<char32_t>(U"", 1, U"\0xx");
  test_strncpy<char32_t>(U"", 2, U"\0\0x");
  test_strncpy<char32_t>(U"", 3, U"\0\0\0");
  test_strncpy<char32_t>(U"a", 0, U"xxx");
  test_strncpy<char32_t>(U"a", 1, U"axx");
  test_strncpy<char32_t>(U"a", 2, U"a\0x");
  test_strncpy<char32_t>(U"a", 3, U"a\0\0");
  test_strncpy<char32_t>(U"\0a", 0, U"xxx");
  test_strncpy<char32_t>(U"\0a", 1, U"\0xx");
  test_strncpy<char32_t>(U"\0a", 2, U"\0\0x");
  test_strncpy<char32_t>(U"\0a", 3, U"\0\0\0");
  test_strncpy<char32_t>(U"hello", 5, U"helloxxx");
  test_strncpy<char32_t>(U"hello", 6, U"hello\0xx");
  test_strncpy<char32_t>(U"hello", 7, U"hello\0\0x");
  test_strncpy<char32_t>(U"hello", 8, U"hello\0\0\0");
  test_strncpy<char32_t>(U"hell\0o", 4, U"hellxxxx");
  test_strncpy<char32_t>(U"hell\0o", 5, U"hell\0xx");
  test_strncpy<char32_t>(U"hell\0o", 6, U"hell\0\0x");
  test_strncpy<char32_t>(U"hell\0o", 7, U"hell\0\0\0");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

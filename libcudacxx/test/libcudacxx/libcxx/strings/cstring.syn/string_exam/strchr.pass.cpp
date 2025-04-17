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
__host__ __device__ constexpr void test_strchr(T* str, T c, T* expected_ret)
{
  const auto ret = cuda::std::__cccl_strchr(str, c);
  assert(ret == expected_ret);
}

template <class T>
__host__ __device__ constexpr void test_type();

#define TEST_SPECIALIZATION(T, P)                   \
  template <>                                       \
  __host__ __device__ constexpr void test_type<T>() \
  {                                                 \
    {                                               \
      T str[]{P##""};                               \
      test_strchr<T>(str, P##'\0', str);            \
      test_strchr<T>(str, P##'a', nullptr);         \
    }                                               \
    {                                               \
      T str[]{P##"a"};                              \
      test_strchr<T>(str, P##'\0', str + 1);        \
      test_strchr<T>(str, P##'a', str);             \
      test_strchr<T>(str, P##'b', nullptr);         \
    }                                               \
    {                                               \
      T str[]{P##"aaa"};                            \
      test_strchr<T>(str, P##'\0', str + 3);        \
      test_strchr<T>(str, P##'a', str);             \
      test_strchr<T>(str, P##'b', nullptr);         \
    }                                               \
    {                                               \
      T str[]{P##"abcdabcd"};                       \
      test_strchr<T>(str, P##'\0', str + 8);        \
      test_strchr<T>(str, P##'a', str);             \
      test_strchr<T>(str, P##'b', str + 1);         \
      test_strchr<T>(str, P##'c', str + 2);         \
      test_strchr<T>(str, P##'d', str + 3);         \
      test_strchr<T>(str, P##'e', nullptr);         \
      test_strchr<T>(str, P##'f', nullptr);         \
    }                                               \
  }

TEST_SPECIALIZATION(char, )
#if _LIBCUDACXX_HAS_CHAR8_T()
TEST_SPECIALIZATION(char8_t, u8)
#endif // _LIBCUDACXX_HAS_CHAR8_T()
TEST_SPECIALIZATION(char16_t, u)
TEST_SPECIALIZATION(char32_t, U)

__host__ __device__ constexpr bool test()
{
  test_type<char>();
#if _LIBCUDACXX_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _LIBCUDACXX_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

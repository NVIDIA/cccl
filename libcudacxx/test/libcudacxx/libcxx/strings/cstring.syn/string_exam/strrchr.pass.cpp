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
__host__ __device__ constexpr void test_strrchr(T* str, T c, T* expected_ret)
{
  const auto ret = cuda::std::__cccl_strrchr(str, c);
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
      test_strrchr<T>(str, P##'\0', str);           \
      test_strrchr<T>(str, P##'a', nullptr);        \
    }                                               \
    {                                               \
      T str[]{P##"a"};                              \
      test_strrchr<T>(str, P##'\0', str + 1);       \
      test_strrchr<T>(str, P##'a', str);            \
      test_strrchr<T>(str, P##'b', nullptr);        \
    }                                               \
    {                                               \
      T str[]{P##"aaa"};                            \
      test_strrchr<T>(str, P##'\0', str + 3);       \
      test_strrchr<T>(str, P##'a', str + 2);        \
      test_strrchr<T>(str, P##'b', nullptr);        \
    }                                               \
    {                                               \
      T str[]{P##"abcdabcd\0\0"};                   \
      test_strrchr<T>(str, P##'\0', str + 8);       \
      test_strrchr<T>(str, P##'a', str + 4);        \
      test_strrchr<T>(str, P##'b', str + 5);        \
      test_strrchr<T>(str, P##'c', str + 6);        \
      test_strrchr<T>(str, P##'d', str + 7);        \
      test_strrchr<T>(str, P##'e', nullptr);        \
      test_strrchr<T>(str, P##'f', nullptr);        \
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

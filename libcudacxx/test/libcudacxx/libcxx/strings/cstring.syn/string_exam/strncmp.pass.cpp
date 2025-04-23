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

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_strncmp(const T* lhs, const T* rhs, cuda::std::size_t n, int expected)
{
  const auto ret = cuda::std::__cccl_strncmp(lhs, rhs, n);

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

template <class T>
__host__ __device__ constexpr void test_type();

#define TEST_SPECIALIZATION(T, P)                       \
  template <>                                           \
  __host__ __device__ constexpr void test_type<T>()     \
  {                                                     \
    test_strncmp<T>(P##"", P##"", 0, 0);                \
    test_strncmp<T>(P##"", P##"", 1, 0);                \
                                                        \
    test_strncmp<T>(P##"a", P##"", 0, 0);               \
    test_strncmp<T>(P##"", P##"a", 0, 0);               \
    test_strncmp<T>(P##"a", P##"", 1, 1);               \
    test_strncmp<T>(P##"", P##"a", 1, -1);              \
                                                        \
    test_strncmp<T>(P##"hi", P##"hi", 0, 0);            \
    test_strncmp<T>(P##"hi", P##"ho", 0, 0);            \
    test_strncmp<T>(P##"ho", P##"hi", 0, 0);            \
                                                        \
    test_strncmp<T>(P##"hi", P##"hi", 1, 0);            \
    test_strncmp<T>(P##"hi", P##"ho", 1, 0);            \
    test_strncmp<T>(P##"ho", P##"hi", 1, 0);            \
                                                        \
    test_strncmp<T>(P##"hi", P##"hi", 2, 0);            \
    test_strncmp<T>(P##"hi", P##"ho", 2, -1);           \
    test_strncmp<T>(P##"ho", P##"hi", 2, 1);            \
                                                        \
    test_strncmp<T>(P##"hi", P##"hi", 3, 0);            \
    test_strncmp<T>(P##"hi", P##"ho", 3, -1);           \
    test_strncmp<T>(P##"ho", P##"hi", 3, 1);            \
                                                        \
    test_strncmp<T>(P##"abcde", P##"abcde", 100, 0);    \
    test_strncmp<T>(P##"abcd1", P##"abcd0", 100, 1);    \
    test_strncmp<T>(P##"abcd0", P##"abcd1", 100, -1);   \
    test_strncmp<T>(P##"ab1de", P##"abcd0", 100, -1);   \
                                                        \
    test_strncmp<T>(P##"abc\0de", P##"abcde", 100, -1); \
    test_strncmp<T>(P##"abc\0d1", P##"abcd0", 100, -1); \
    test_strncmp<T>(P##"abc\0d0", P##"abcd1", 100, -1); \
    test_strncmp<T>(P##"ab1\0de", P##"abcd0", 100, -1); \
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

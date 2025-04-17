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
__host__ __device__ constexpr bool test_out_of_place();

template <class T>
__host__ __device__ constexpr bool test_in_place();

#define TEST_SPECIALIZATION(T, P)                                          \
  template <>                                                              \
  __host__ __device__ constexpr bool test_out_of_place<T>()                \
  {                                                                        \
    T src[] = P##"1234567890";                                             \
                                                                           \
    {                                                                      \
      T dest[] = P##"abcdefghij";                                          \
      T ref[]  = P##"1234567890";                                          \
      assert(cuda::std::__cccl_memmove(dest, src, 10) == dest);            \
      assert(cuda::std::__cccl_memcmp(dest, ref, 10) == 0);                \
    }                                                                      \
                                                                           \
    {                                                                      \
      T dest[] = P##"abcdefghij";                                          \
      T ref[]  = P##"abc123456j";                                          \
      assert(cuda::std::__cccl_memmove(dest + 3, src, 6) == dest + 3);     \
      assert(cuda::std::__cccl_memcmp(dest, ref, 10) == 0);                \
    }                                                                      \
                                                                           \
    {                                                                      \
      T dest[] = P##"abcdefghij";                                          \
      T ref[]  = P##"56789fghij";                                          \
      assert(cuda::std::__cccl_memmove(dest, src + 4, 5) == dest);         \
      assert(cuda::std::__cccl_memcmp(dest, ref, 10) == 0);                \
    }                                                                      \
                                                                           \
    {                                                                      \
      T dest[] = P##"abcdefghij";                                          \
      T ref[]  = P##"ab789fghij";                                          \
      assert(cuda::std::__cccl_memmove(dest + 2, src + 6, 3) == dest + 2); \
      assert(cuda::std::__cccl_memcmp(dest, ref, 10) == 0);                \
    }                                                                      \
                                                                           \
    return true;                                                           \
  }                                                                        \
                                                                           \
  template <>                                                              \
  __host__ __device__ bool test_in_place<T>()                              \
  {                                                                        \
    {                                                                      \
      T buf[] = P##"1234567890";                                           \
      T ref[] = P##"1234567890";                                           \
      assert(cuda::std::__cccl_memmove(buf, buf, 10) == buf);              \
      assert(cuda::std::__cccl_memcmp(buf, ref, 10) == 0);                 \
    }                                                                      \
                                                                           \
    {                                                                      \
      T buf[] = P##"1234567890";                                           \
      T ref[] = P##"1231234567";                                           \
      assert(cuda::std::__cccl_memmove(buf + 3, buf, 7) == buf + 3);       \
      assert(cuda::std::__cccl_memcmp(buf, ref, 10) == 0);                 \
    }                                                                      \
                                                                           \
    {                                                                      \
      T buf[] = P##"1234567890";                                           \
      T ref[] = P##"5678967890";                                           \
      assert(cuda::std::__cccl_memmove(buf, buf + 4, 5) == buf);           \
      assert(cuda::std::__cccl_memcmp(buf, ref, 10) == 0);                 \
    }                                                                      \
                                                                           \
    {                                                                      \
      T buf[] = P##"1234567890";                                           \
      T ref[] = P##"1234897890";                                           \
      assert(cuda::std::__cccl_memmove(buf + 4, buf + 7, 2) == buf + 4);   \
      assert(cuda::std::__cccl_memcmp(buf, ref, 10) == 0);                 \
    }                                                                      \
                                                                           \
    return true;                                                           \
  }

TEST_SPECIALIZATION(char, )
#if _LIBCUDACXX_HAS_CHAR8_T()
TEST_SPECIALIZATION(char8_t, u8)
#endif // _LIBCUDACXX_HAS_CHAR8_T()
TEST_SPECIALIZATION(char16_t, u)
TEST_SPECIALIZATION(char32_t, U)

__host__ __device__ bool test()
{
  test_out_of_place<char>();
#if _LIBCUDACXX_HAS_CHAR8_T()
  test_out_of_place<char8_t>();
#endif // _LIBCUDACXX_HAS_CHAR8_T()
  test_out_of_place<char16_t>();
  test_out_of_place<char32_t>();

  test_in_place<char>();
#if _LIBCUDACXX_HAS_CHAR8_T()
  test_in_place<char8_t>();
#endif // _LIBCUDACXX_HAS_CHAR8_T()
  test_in_place<char16_t>();
  // test_in_place<char32_t>(); // crashing on device

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}

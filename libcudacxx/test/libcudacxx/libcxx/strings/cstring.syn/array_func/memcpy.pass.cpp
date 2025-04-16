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

template <class T, size_t n>
__host__ __device__ constexpr void test_memcpy(const T* src)
{
  T buf[n + 1]{}; // + 1 to prevent zero size buffer
  assert(cuda::std::__cccl_memcpy(buf, src, n) == buf);
  assert(cuda::std::__cccl_memcmp(buf, src, n) == 0);
}

__host__ __device__ constexpr bool test()
{
  // char
  test_memcpy<char, 0>("");
  test_memcpy<char, 0>("asdf");
  test_memcpy<char, 1>("a");
  test_memcpy<char, 3>("abcde");
  test_memcpy<char, 5>("abcde");

#if _LIBCUDACXX_HAS_CHAR8_T()
  // char8_t
  test_memcpy<char8_t, 0>(u8"");
  test_memcpy<char8_t, 0>(u8"asdf");
  test_memcpy<char8_t, 1>(u8"a");
  test_memcpy<char8_t, 3>(u8"abcde");
  test_memcpy<char8_t, 5>(u8"abcde");
#endif // _LIBCUDACXX_HAS_CHAR8_T()

  // char16_t
  test_memcpy<char16_t, 0>(u"");
  test_memcpy<char16_t, 0>(u"asdf");
  test_memcpy<char16_t, 1>(u"a");
  test_memcpy<char16_t, 3>(u"abcde");
  test_memcpy<char16_t, 5>(u"abcde");

  // char32_t
  test_memcpy<char32_t, 0>(U"");
  test_memcpy<char32_t, 0>(U"asdf");
  test_memcpy<char32_t, 1>(U"a");
  test_memcpy<char32_t, 3>(U"abcde");
  test_memcpy<char32_t, 5>(U"abcde");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

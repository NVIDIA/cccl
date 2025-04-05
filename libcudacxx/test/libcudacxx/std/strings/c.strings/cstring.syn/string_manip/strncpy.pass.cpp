//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <cuda::std::size_t... N>
__host__ __device__ constexpr bool equal_buffers(const char* lhs, const char* rhs, cuda::std::index_sequence<N...>)
{
  return ((lhs[N] == rhs[N]) && ...);
}

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_strncpy(const char* str, cuda::std::size_t count, const char (&ref)[N])
{
  char buff[N]{};
  for (cuda::std::size_t i = 0; i < N; ++i)
  {
    buff[i] = 'x';
  }

  const auto ret = cuda::std::strncpy(buff, str, count);
  assert(ret == buff);
  assert(equal_buffers(buff, ref, cuda::std::make_index_sequence<N - 1>{}));
}

__host__ __device__ constexpr bool test()
{
  static_assert(
    cuda::std::is_same_v<char*,
                         decltype(cuda::std::strncpy(
                           cuda::std::declval<char*>(), cuda::std::declval<const char*>(), cuda::std::size_t{}))>);

  test_strncpy("", 0, "xxx");
  test_strncpy("", 1, "\0xx");
  test_strncpy("", 2, "\0\0x");
  test_strncpy("", 3, "\0\0\0");

  test_strncpy("a", 0, "xxx");
  test_strncpy("a", 1, "axx");
  test_strncpy("a", 2, "a\0x");
  test_strncpy("a", 3, "a\0\0");

  test_strncpy("\0a", 0, "xxx");
  test_strncpy("\0a", 1, "\0xx");
  test_strncpy("\0a", 2, "\0\0x");
  test_strncpy("\0a", 3, "\0\0\0");

  test_strncpy("hello", 5, "helloxxx");
  test_strncpy("hello", 6, "hello\0xx");
  test_strncpy("hello", 7, "hello\0\0x");
  test_strncpy("hello", 8, "hello\0\0\0");

  test_strncpy("hell\0o", 4, "hellxxxx");
  test_strncpy("hell\0o", 5, "hell\0xx");
  test_strncpy("hell\0o", 6, "hell\0\0x");
  test_strncpy("hell\0o", 7, "hell\0\0\0");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

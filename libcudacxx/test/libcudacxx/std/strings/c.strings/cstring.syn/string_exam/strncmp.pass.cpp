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

__host__ __device__ constexpr void test_strncmp(const char* lhs, const char* rhs, cuda::std::size_t n, int expected)
{
  const auto ret = cuda::std::strncmp(lhs, rhs, n);

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
  static_assert(
    cuda::std::is_same_v<int,
                         decltype(cuda::std::strncmp(
                           cuda::std::declval<const char*>(), cuda::std::declval<const char*>(), cuda::std::size_t{}))>);

  test_strncmp("", "", 0, 0);
  test_strncmp("", "", 1, 0);

  test_strncmp("a", "", 0, 0);
  test_strncmp("", "a", 0, 0);
  test_strncmp("a", "", 1, 1);
  test_strncmp("", "a", 1, -1);

  test_strncmp("hi", "hi", 0, 0);
  test_strncmp("hi", "ho", 0, 0);
  test_strncmp("ho", "hi", 0, 0);

  test_strncmp("hi", "hi", 1, 0);
  test_strncmp("hi", "ho", 1, 0);
  test_strncmp("ho", "hi", 1, 0);

  test_strncmp("hi", "hi", 2, 0);
  test_strncmp("hi", "ho", 2, -1);
  test_strncmp("ho", "hi", 2, 1);

  test_strncmp("hi", "hi", 3, 0);
  test_strncmp("hi", "ho", 3, -1);
  test_strncmp("ho", "hi", 3, 1);

  test_strncmp("abcde", "abcde", 100, 0);
  test_strncmp("abcd1", "abcd0", 100, 1);
  test_strncmp("abcd0", "abcd1", 100, -1);
  test_strncmp("ab1de", "abcd0", 100, -1);

  test_strncmp("abc\0de", "abcde", 100, -1);
  test_strncmp("abc\0d1", "abcd0", 100, -1);
  test_strncmp("abc\0d0", "abcd1", 100, -1);
  test_strncmp("ab1\0de", "abcd0", 100, -1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

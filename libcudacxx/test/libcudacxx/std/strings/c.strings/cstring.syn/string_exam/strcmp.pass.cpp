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

__host__ __device__ constexpr void test_strcmp(const char* lhs, const char* rhs, int expected)
{
  const auto ret = cuda::std::strcmp(lhs, rhs);

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
    cuda::std::
      is_same_v<int, decltype(cuda::std::strcmp(cuda::std::declval<const char*>(), cuda::std::declval<const char*>()))>);

  test_strcmp("", "", 0);

  test_strcmp("a", "", 1);
  test_strcmp("", "a", -1);

  test_strcmp("hi", "hi", 0);
  test_strcmp("hi", "ho", -1);
  test_strcmp("ho", "hi", 1);

  test_strcmp("abcde", "abcde", 0);
  test_strcmp("abcd1", "abcd0", 1);
  test_strcmp("abcd0", "abcd1", -1);
  test_strcmp("ab1de", "abcd0", -1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

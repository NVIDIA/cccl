//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// int memcmp(const void* lhs, const void* rhs, size_t count);

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>

#include "test_macros.h"

__host__ __device__ void test_memcmp(const char* lhs, const char* rhs, size_t n, int expected)
{
  const auto ret = cuda::std::memcmp(lhs, rhs, n);

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

__host__ __device__ bool test()
{
  test_memcmp("abcde", "abcde", 5, 0);
  test_memcmp("abcd1", "abcd0", 5, 1);
  test_memcmp("abcd0", "abcd1", 5, -1);

  test_memcmp("abcd1", "abcd0", 4, 0);
  test_memcmp("abcd0", "abcd1", 4, 0);

  test_memcmp("abcde", "fghij", 5, -1);
  test_memcmp("abcde", "fghij", 0, 0);

  test_memcmp(nullptr, nullptr, 0, 0);

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}

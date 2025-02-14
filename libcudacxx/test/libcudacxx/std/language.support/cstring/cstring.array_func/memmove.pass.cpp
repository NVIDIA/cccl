//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// void* memmove(void* dst, const void* src, size_t count);

#include <cuda/std/cassert>
#include <cuda/std/cstring>

#include "test_macros.h"

__host__ __device__ bool test_out_of_place()
{
  char src[] = "1234567890";

  {
    char dest[] = "abcdefghij";
    char ref[]  = "1234567890";
    assert(cuda::std::memmove(dest, src, 10) == dest);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  {
    char dest[] = "abcdefghij";
    char ref[]  = "abc123456j";
    assert(cuda::std::memmove(dest + 3, src, 6) == dest + 3);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  {
    char dest[] = "abcdefghij";
    char ref[]  = "56789fghij";
    assert(cuda::std::memmove(dest, src + 4, 5) == dest);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  {
    char dest[] = "abcdefghij";
    char ref[]  = "ab789fghij";
    assert(cuda::std::memmove(dest + 2, src + 6, 3) == dest + 2);
    assert(cuda::std::memcmp(dest, ref, 10) == 0);
  }

  return true;
}

__host__ __device__ bool test_in_place()
{
  {
    char buf[] = "1234567890";
    char ref[] = "1234567890";
    assert(cuda::std::memmove(buf, buf, 10) == buf);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  {
    char buf[] = "1234567890";
    char ref[] = "1231234567";
    assert(cuda::std::memmove(buf + 3, buf, 7) == buf + 3);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  {
    char buf[] = "1234567890";
    char ref[] = "5678967890";
    assert(cuda::std::memmove(buf, buf + 4, 5) == buf);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  {
    char buf[] = "1234567890";
    char ref[] = "1234897890";
    assert(cuda::std::memmove(buf + 4, buf + 7, 2) == buf + 4);
    assert(cuda::std::memcmp(buf, ref, 10) == 0);
  }

  return true;
}

__host__ __device__ bool test()
{
  test_out_of_place();
  test_in_place();
  return true;
}

int main(int, char**)
{
  test();
  return 0;
}

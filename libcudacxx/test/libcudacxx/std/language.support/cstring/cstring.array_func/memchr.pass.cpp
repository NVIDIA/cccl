//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstring>
#include <cuda/std/type_traits>

constexpr int not_found = -1;

__host__ __device__ void test(const char* ptr, int c, size_t n, int expected_pos)
{
  const void* ret = cuda::std::memchr(ptr, c, n);

  if (expected_pos == not_found)
  {
    assert(ret == nullptr);
  }
  else
  {
    assert(ret != nullptr);
    assert(static_cast<const char*>(ret) == ptr + expected_pos);
  }
}

int main(int, char**)
{
  test("abcde", '\0', 6, 5);
  test("abcde", '\0', 5, not_found);
  test("aaabb", 'b', 5, 3);
  test("aaabb", 'b', 4, 3);
  test("aaabb", 'b', 3, not_found);
  test("aaaa", 'b', 4, not_found);
  test("aaaa", 'a', 0, not_found);

  return 0;
}

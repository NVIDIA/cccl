//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string_>
#include <cuda/std/cassert>

__host__ __device__ constexpr bool test()
{
  char s1[] = {1, 2, 3};
  assert(cuda::std::char_traits<char>::find(s1, 3, char(1)) == s1);
  assert(cuda::std::char_traits<char>::find(s1, 3, char(2)) == s1 + 1);
  assert(cuda::std::char_traits<char>::find(s1, 3, char(3)) == s1 + 2);
  assert(cuda::std::char_traits<char>::find(s1, 3, char(4)) == 0);
  assert(cuda::std::char_traits<char>::find(s1, 3, char(0)) == 0);
  assert(cuda::std::char_traits<char>::find(nullptr, 0, char(0)) == 0);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

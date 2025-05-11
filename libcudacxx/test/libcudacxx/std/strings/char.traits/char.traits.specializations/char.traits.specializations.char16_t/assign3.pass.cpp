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
  char16_t s2[3] = {0};
  assert(cuda::std::char_traits<char16_t>::assign(s2, 3, char16_t(5)) == s2);
  assert(s2[0] == char16_t(5));
  assert(s2[1] == char16_t(5));
  assert(s2[2] == char16_t(5));
  assert(cuda::std::char_traits<char16_t>::assign(nullptr, 0, char16_t(5)) == nullptr);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

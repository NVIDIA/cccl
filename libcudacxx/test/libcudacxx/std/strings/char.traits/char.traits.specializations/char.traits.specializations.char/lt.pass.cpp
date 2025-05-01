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
  assert(cuda::std::char_traits<char>::lt('\0', 'A') == ('\0' < 'A'));
  assert(cuda::std::char_traits<char>::lt('A', '\0') == ('A' < '\0'));

  assert(cuda::std::char_traits<char>::lt('a', 'a') == ('a' < 'a'));
  assert(cuda::std::char_traits<char>::lt('A', 'a') == ('A' < 'a'));
  assert(cuda::std::char_traits<char>::lt('a', 'A') == ('a' < 'A'));

  assert(cuda::std::char_traits<char>::lt('a', 'z') == ('a' < 'z'));
  assert(cuda::std::char_traits<char>::lt('A', 'Z') == ('A' < 'Z'));

  assert(cuda::std::char_traits<char>::lt(' ', 'A') == (' ' < 'A'));
  assert(cuda::std::char_traits<char>::lt('A', '~') == ('A' < '~'));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

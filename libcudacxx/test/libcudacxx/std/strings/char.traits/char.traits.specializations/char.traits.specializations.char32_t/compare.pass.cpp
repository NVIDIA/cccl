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
  assert(cuda::std::char_traits<char32_t>::compare(U"", U"", 0) == 0);
  assert(cuda::std::char_traits<char32_t>::compare(nullptr, nullptr, 0) == 0);

  assert(cuda::std::char_traits<char32_t>::compare(U"1", U"1", 1) == 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"1", U"2", 1) < 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"2", U"1", 1) > 0);

  assert(cuda::std::char_traits<char32_t>::compare(U"12", U"12", 2) == 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"12", U"13", 2) < 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"12", U"22", 2) < 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"13", U"12", 2) > 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"22", U"12", 2) > 0);

  assert(cuda::std::char_traits<char32_t>::compare(U"123", U"123", 3) == 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"123", U"223", 3) < 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"123", U"133", 3) < 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"123", U"124", 3) < 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"223", U"123", 3) > 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"133", U"123", 3) > 0);
  assert(cuda::std::char_traits<char32_t>::compare(U"124", U"123", 3) > 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

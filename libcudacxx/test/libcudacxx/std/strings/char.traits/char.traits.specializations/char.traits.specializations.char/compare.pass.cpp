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
  assert(cuda::std::char_traits<char>::compare("", "", 0) == 0);
  assert(cuda::std::char_traits<char>::compare(nullptr, nullptr, 0) == 0);

  assert(cuda::std::char_traits<char>::compare("1", "1", 1) == 0);
  assert(cuda::std::char_traits<char>::compare("1", "2", 1) < 0);
  assert(cuda::std::char_traits<char>::compare("2", "1", 1) > 0);

  assert(cuda::std::char_traits<char>::compare("12", "12", 2) == 0);
  assert(cuda::std::char_traits<char>::compare("12", "13", 2) < 0);
  assert(cuda::std::char_traits<char>::compare("12", "22", 2) < 0);
  assert(cuda::std::char_traits<char>::compare("13", "12", 2) > 0);
  assert(cuda::std::char_traits<char>::compare("22", "12", 2) > 0);

  assert(cuda::std::char_traits<char>::compare("123", "123", 3) == 0);
  assert(cuda::std::char_traits<char>::compare("123", "223", 3) < 0);
  assert(cuda::std::char_traits<char>::compare("123", "133", 3) < 0);
  assert(cuda::std::char_traits<char>::compare("123", "124", 3) < 0);
  assert(cuda::std::char_traits<char>::compare("223", "123", 3) > 0);
  assert(cuda::std::char_traits<char>::compare("133", "123", 3) > 0);
  assert(cuda::std::char_traits<char>::compare("124", "123", 3) > 0);

  {
    char a[] = {static_cast<char>(-1), 0};
    char b[] = {1, 0};
    assert(cuda::std::char_traits<char>::compare(a, b, 1) > 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

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
  assert(cuda::std::char_traits<char16_t>::compare(u"", u"", 0) == 0);
  assert(cuda::std::char_traits<char16_t>::compare(NULL, NULL, 0) == 0);

  assert(cuda::std::char_traits<char16_t>::compare(u"1", u"1", 1) == 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"1", u"2", 1) < 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"2", u"1", 1) > 0);

  assert(cuda::std::char_traits<char16_t>::compare(u"12", u"12", 2) == 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"12", u"13", 2) < 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"12", u"22", 2) < 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"13", u"12", 2) > 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"22", u"12", 2) > 0);

  assert(cuda::std::char_traits<char16_t>::compare(u"123", u"123", 3) == 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"123", u"223", 3) < 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"123", u"133", 3) < 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"123", u"124", 3) < 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"223", u"123", 3) > 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"133", u"123", 3) > 0);
  assert(cuda::std::char_traits<char16_t>::compare(u"124", u"123", 3) > 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

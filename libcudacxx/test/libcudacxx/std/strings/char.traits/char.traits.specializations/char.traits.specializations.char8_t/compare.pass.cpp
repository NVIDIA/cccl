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
#if _CCCL_HAS_CHAR8_T()
  assert(cuda::std::char_traits<char8_t>::compare(u8"", u8"", 0) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(nullptr, nullptr, 0) == 0);

  assert(cuda::std::char_traits<char8_t>::compare(u8"1", u8"1", 1) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"1", u8"2", 1) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"2", u8"1", 1) > 0);

  assert(cuda::std::char_traits<char8_t>::compare(u8"12", u8"12", 2) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"12", u8"13", 2) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"12", u8"22", 2) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"13", u8"12", 2) > 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"22", u8"12", 2) > 0);

  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"123", 3) == 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"223", 3) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"133", 3) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"123", u8"124", 3) < 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"223", u8"123", 3) > 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"133", u8"123", 3) > 0);
  assert(cuda::std::char_traits<char8_t>::compare(u8"124", u8"123", 3) > 0);
#endif // _CCCL_HAS_CHAR8_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

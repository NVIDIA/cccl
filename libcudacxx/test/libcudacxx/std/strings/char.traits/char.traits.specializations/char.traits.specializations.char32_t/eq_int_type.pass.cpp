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
  assert(cuda::std::char_traits<char32_t>::eq_int_type(U'a', U'a'));
  assert(!cuda::std::char_traits<char32_t>::eq_int_type(U'a', U'A'));
  // assert(!cuda::std::char_traits<char32_t>::eq_int_type(cuda::std::char_traits<char32_t>::eof(), U'A'));
  // assert(cuda::std::char_traits<char32_t>::eq_int_type(
  //   cuda::std::char_traits<char32_t>::eof(), cuda::std::char_traits<char32_t>::eof()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

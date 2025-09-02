//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

#include <cuda/std/__string_>
#include <cuda/std/cassert>

__host__ __device__ constexpr bool test()
{
  assert(cuda::std::char_traits<char>::not_eof('a') == 'a');
  assert(cuda::std::char_traits<char>::not_eof('A') == 'A');
  assert(cuda::std::char_traits<char>::not_eof(0) == 0);
  assert(cuda::std::char_traits<char>::not_eof(cuda::std::char_traits<char>::eof())
         != cuda::std::char_traits<char>::eof());

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

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
  assert(cuda::std::char_traits<char16_t>::length(u"") == 0);
  assert(cuda::std::char_traits<char16_t>::length(u"a") == 1);
  assert(cuda::std::char_traits<char16_t>::length(u"aa") == 2);
  assert(cuda::std::char_traits<char16_t>::length(u"aaa") == 3);
  assert(cuda::std::char_traits<char16_t>::length(u"aaaa") == 4);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}

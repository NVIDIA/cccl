//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 12) // older gcc cannot determine range<decltype(jv)>
  cuda::std::ranges::sentinel_t<cuda::std::ranges::join_view<CopyableParent>> sent;
#else // ^^^ gcc < 12 ^^^ / vvv  !(gcc < 12) vvv
  using sentinel = decltype(cuda::std::declval<cuda::std::ranges::join_view<CopyableParent>&>().end());
  sentinel sent;
#endif // !(gcc < 12)
  unused(sent);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

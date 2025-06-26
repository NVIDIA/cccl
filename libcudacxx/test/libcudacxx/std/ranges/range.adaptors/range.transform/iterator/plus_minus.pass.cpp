//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_view::<iterator>::operator{+,-}

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable> transformView1{};
  auto iter1 = cuda::std::move(transformView1).begin();
  cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable> transformView2{};
  auto iter2 = cuda::std::move(transformView2).begin();
  iter1 += 4;
  assert((iter1 + 1).base() == globalBuff + 5);
  assert((1 + iter1).base() == globalBuff + 5);
  assert((iter1 - 1).base() == globalBuff + 3);
  assert(iter1 - iter2 == 4);
  assert((iter1 + 2) - 2 == iter1);
  assert((iter1 - 2) + 2 == iter1);

  unused(iter2);
  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}

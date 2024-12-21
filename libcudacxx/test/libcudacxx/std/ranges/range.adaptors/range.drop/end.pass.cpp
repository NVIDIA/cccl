//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr auto end()
//   requires (!simple-view<V>)
// constexpr auto end() const
//   requires range<const V>

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  // range<const V>
  cuda::std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.end() == globalBuff + 8);

  // !simple-view<V>
  cuda::std::ranges::drop_view dropView2(InputView(), 4);
  assert(dropView2.end() == globalBuff + 8);

  // range<const V>
  const cuda::std::ranges::drop_view dropView3(MoveOnlyView(), 0);
  assert(dropView3.end() == globalBuff + 8);

  // !simple-view<V>
  const cuda::std::ranges::drop_view dropView4(InputView(), 2);
  assert(dropView4.end() == globalBuff + 8);

  // range<const V>
  cuda::std::ranges::drop_view dropView5(MoveOnlyView(), 10);
  assert(dropView5.end() == globalBuff + 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

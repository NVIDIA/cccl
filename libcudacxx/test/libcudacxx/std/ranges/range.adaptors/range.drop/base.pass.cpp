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

// constexpr V base() const& requires copy_constructible<V> { return base_; }
// constexpr V base() && { return cuda::std::move(base_); }

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::drop_view<MoveOnlyView> dropView1;
  auto base1 = cuda::std::move(dropView1).base();
  assert(cuda::std::ranges::begin(base1) == globalBuff);

  // Note: we should *not* drop two elements here.
  cuda::std::ranges::drop_view<MoveOnlyView> dropView2(MoveOnlyView{4}, 2);
  auto base2 = cuda::std::move(dropView2).base();
  assert(cuda::std::ranges::begin(base2) == globalBuff + 4);

  cuda::std::ranges::drop_view<CopyableView> dropView3;
  auto base3 = dropView3.base();
  assert(cuda::std::ranges::begin(base3) == globalBuff);
  auto base4 = cuda::std::move(dropView3).base();
  assert(cuda::std::ranges::begin(base4) == globalBuff);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

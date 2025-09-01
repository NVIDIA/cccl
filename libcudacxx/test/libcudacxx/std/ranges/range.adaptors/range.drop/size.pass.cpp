//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto size()
//   requires sized_range<V>
// constexpr auto size() const
//   requires sized_range<const V>

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT SizeInvocable = _CCCL_REQUIRES_EXPR((T), cuda::std::ranges::drop_view<T> v)((v.size()));

__host__ __device__ constexpr bool test()
{
  // sized_range<V>
  cuda::std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.size() == 4);

  // sized_range<V>
  cuda::std::ranges::drop_view dropView2(MoveOnlyView(), 0);
  assert(dropView2.size() == 8);

  // sized_range<const V>
  const cuda::std::ranges::drop_view dropView3(MoveOnlyView(), 8);
  assert(dropView3.size() == 0);

  // sized_range<const V>
  const cuda::std::ranges::drop_view dropView4(MoveOnlyView(), 10);
  assert(dropView4.size() == 0);

  // Because ForwardView is not a sized_range.
  static_assert(!SizeInvocable<ForwardView>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

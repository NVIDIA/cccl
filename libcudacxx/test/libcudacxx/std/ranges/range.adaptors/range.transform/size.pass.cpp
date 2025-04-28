//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT SizeInvocable = _CCCL_REQUIRES_EXPR((T), T t)(((void) t.size()));

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{}, PlusOne{});
    assert(transformView.size() == 8);
  }

  {
    const cuda::std::ranges::transform_view transformView(MoveOnlyView{globalBuff, 4}, PlusOne{});
    assert(transformView.size() == 4);
  }

  static_assert(!SizeInvocable<cuda::std::ranges::transform_view<ForwardView, PlusOne>>);

  static_assert(SizeInvocable<cuda::std::ranges::transform_view<SizedSentinelNotConstView, PlusOne>>);
  static_assert(!SizeInvocable<const cuda::std::ranges::transform_view<SizedSentinelNotConstView, PlusOne>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

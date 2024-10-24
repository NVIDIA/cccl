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

// constexpr V base() const& requires copy_constructible<V>
// constexpr V base() &&

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::transform_view<MoveOnlyView, PlusOne> transformView{};
    MoveOnlyView base = cuda::std::move(transformView).base();
    ASSERT_SAME_TYPE(MoveOnlyView, decltype(cuda::std::move(transformView).base()));
    assert(cuda::std::ranges::begin(base) == globalBuff);
  }

  {
    cuda::std::ranges::transform_view<CopyableView, PlusOne> transformView{};
    CopyableView base1 = transformView.base();
    ASSERT_SAME_TYPE(CopyableView, decltype(transformView.base()));
    assert(cuda::std::ranges::begin(base1) == globalBuff);

    CopyableView base2 = cuda::std::move(transformView).base();
    ASSERT_SAME_TYPE(CopyableView, decltype(cuda::std::move(transformView).base()));
    assert(cuda::std::ranges::begin(base2) == globalBuff);
  }

  {
    const cuda::std::ranges::transform_view<CopyableView, PlusOne> transformView{};
    const CopyableView base1 = transformView.base();
    ASSERT_SAME_TYPE(CopyableView, decltype(transformView.base()));
    assert(cuda::std::ranges::begin(base1) == globalBuff);

    const CopyableView base2 = cuda::std::move(transformView).base();
    ASSERT_SAME_TYPE(CopyableView, decltype(cuda::std::move(transformView).base()));
    assert(cuda::std::ranges::begin(base2) == globalBuff);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

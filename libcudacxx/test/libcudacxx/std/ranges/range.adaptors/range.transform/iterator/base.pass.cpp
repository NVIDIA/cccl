//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_view::<iterator>::base

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    using TransformView = cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    TransformView tv{};
    auto it  = tv.begin();
    using It = decltype(it);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&>(it).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&&>(it).base()), int*>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&>(it).base()), int* const&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&&>(it).base()), int* const&>);
    static_assert(noexcept(it.base()));
    assert(base(it.base()) == globalBuff);
    assert(base(cuda::std::move(it).base()) == globalBuff);
  }
  {
    using TransformView = cuda::std::ranges::transform_view<InputView, PlusOneMutable>;
    TransformView tv{};
    auto it  = tv.begin();
    using It = decltype(it);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&>(it).base()), const cpp20_input_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<It&&>(it).base()), cpp20_input_iterator<int*>>);
    static_assert(cuda::std::is_same_v<decltype(static_cast<const It&>(it).base()), const cpp20_input_iterator<int*>&>);
    static_assert(
      cuda::std::is_same_v<decltype(static_cast<const It&&>(it).base()), const cpp20_input_iterator<int*>&>);
    static_assert(noexcept(it.base()));
    assert(base(it.base()) == globalBuff);
    assert(base(cuda::std::move(it).base()) == globalBuff);
  }
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

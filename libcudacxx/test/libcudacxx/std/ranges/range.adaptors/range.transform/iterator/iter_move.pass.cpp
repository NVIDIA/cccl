//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr decltype(auto) iter_move(const iterator& i)
//    noexcept(noexcept(invoke(i.parent_->fun_, *i.current_)))

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    auto iter = transformView.begin();

    static_assert(!noexcept(cuda::std::ranges::iter_move(iter)));

    assert(cuda::std::ranges::iter_move(iter) == 1);
    assert(cuda::std::ranges::iter_move(iter + 2) == 3);

    static_assert(cuda::std::is_same_v<int, decltype(cuda::std::ranges::iter_move(iter))>);
    static_assert(cuda::std::is_same_v<int, decltype(cuda::std::ranges::iter_move(cuda::std::move(iter)))>);
  }

  {
    static_assert(noexcept(cuda::std::ranges::iter_move(
      cuda::std::declval<
        cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOneNoexcept>>&>())));

    static_assert(!noexcept(cuda::std::ranges::iter_move(
      cuda::std::declval<
        cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>&>())));
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

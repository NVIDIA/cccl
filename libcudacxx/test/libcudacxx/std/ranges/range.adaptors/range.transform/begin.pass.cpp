//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator<false> begin();
// constexpr iterator<true> begin() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT BeginInvocable = _CCCL_REQUIRES_EXPR((T), T t)((t.begin()));

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    assert(transformView.begin().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    cuda::std::ranges::transform_view transformView(ForwardView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    cuda::std::ranges::transform_view transformView(InputView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    const cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
  }

  static_assert(!BeginInvocable<const cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);

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

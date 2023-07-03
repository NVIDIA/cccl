//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// transform_view::<iterator>::operator*

#include <cuda/std/ranges>

#include "test_macros.h"
#include "../types.h"

int main(int, char**) {
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
    View transformView(MoveOnlyView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
    ASSERT_SAME_TYPE(int, decltype(*cuda::std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    View transformView(MoveOnlyView{buff}, PlusOneMutable{});
    assert(*transformView.begin() == 1);
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
    ASSERT_SAME_TYPE(int, decltype(*cuda::std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, PlusOneNoexcept>;
    View transformView(MoveOnlyView{buff}, PlusOneNoexcept{});
    assert(*transformView.begin() == 1);
    LIBCPP_ASSERT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
    ASSERT_SAME_TYPE(int, decltype(*cuda::std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, Increment>;
    View transformView(MoveOnlyView{buff}, Increment{});
    assert(*transformView.begin() == 1);
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
    ASSERT_SAME_TYPE(int&, decltype(*cuda::std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, IncrementRvalueRef>;
    View transformView(MoveOnlyView{buff}, IncrementRvalueRef{});
    assert(*transformView.begin() == 1);
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
    ASSERT_SAME_TYPE(int&&, decltype(*cuda::std::declval<View>().begin()));
  }

  return 0;
}

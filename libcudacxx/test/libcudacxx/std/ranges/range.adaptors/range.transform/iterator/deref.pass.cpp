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

// transform_view::<iterator>::operator*

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>;
    View transformView(MoveOnlyView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(int, decltype(*cuda::std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    View transformView(MoveOnlyView{buff}, PlusOneMutable{});
    assert(*transformView.begin() == 1);
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
#endif // !TEST_COMPILER_ICC
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
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(int&, decltype(*cuda::std::declval<View>().begin()));
  }
  {
    int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};
    using View = cuda::std::ranges::transform_view<MoveOnlyView, IncrementRvalueRef>;
    View transformView(MoveOnlyView{buff}, IncrementRvalueRef{});
    assert(*transformView.begin() == 1);
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(*cuda::std::declval<cuda::std::ranges::iterator_t<View>>());
    ASSERT_SAME_TYPE(int&&, decltype(*cuda::std::declval<View>().begin()));
#endif // !TEST_COMPILER_ICC
  }

  return 0;
}

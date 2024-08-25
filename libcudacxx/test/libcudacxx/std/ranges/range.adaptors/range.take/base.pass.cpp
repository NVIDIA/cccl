//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr V base() const& requires copy_constructible<V>;
// constexpr V base() &&;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class View, class = void>
inline constexpr bool HasBase = false;

template <class View>
inline constexpr bool HasBase<View, cuda::std::void_t<decltype(cuda::std::declval<View>().base())>> = true;

template <class View>
__host__ __device__ constexpr bool hasLValueQualifiedBase(View&& view)
{
  return HasBase<decltype(view)>;
}

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 0);
    assert(tv.base().ptr_ == buffer);
    assert(cuda::std::move(tv).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(tv.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(tv).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(tv));
  }

  {
    cuda::std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 1);
    assert(cuda::std::move(tv).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(cuda::std::move(tv).base()), MoveOnlyView);
    static_assert(!hasLValueQualifiedBase(tv));
  }

  {
    const cuda::std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 2);
    assert(tv.base().ptr_ == buffer);
    assert(cuda::std::move(tv).base().ptr_ == buffer);

    ASSERT_SAME_TYPE(decltype(tv.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(tv).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(tv));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

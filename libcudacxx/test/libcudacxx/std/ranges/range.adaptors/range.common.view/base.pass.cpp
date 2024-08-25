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
#include <cuda/std/utility>

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
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    CopyableView view{buf, buf + 8};
    cuda::std::ranges::common_view<CopyableView> common(view);
    assert(common.base().begin_ == buf);
    assert(cuda::std::move(common).base().begin_ == buf);

    ASSERT_SAME_TYPE(decltype(common.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(common).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(common));
  }

  {
    MoveOnlyView view{buf, buf + 8};
    cuda::std::ranges::common_view<MoveOnlyView> common(cuda::std::move(view));
    assert(cuda::std::move(common).base().begin_ == buf);

    ASSERT_SAME_TYPE(decltype(cuda::std::move(common).base()), MoveOnlyView);
    static_assert(!hasLValueQualifiedBase(common));
  }

  {
    CopyableView view{buf, buf + 8};
    const cuda::std::ranges::common_view<CopyableView> common(view);
    assert(common.base().begin_ == buf);
    assert(cuda::std::move(common).base().begin_ == buf);

    ASSERT_SAME_TYPE(decltype(common.base()), CopyableView);
    ASSERT_SAME_TYPE(decltype(cuda::std::move(common).base()), CopyableView);
    static_assert(hasLValueQualifiedBase(common));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

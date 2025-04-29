//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator--() requires decrementable<W>;
// constexpr iterator operator--(int) requires decrementable<W>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

template <class T>
_CCCL_CONCEPT Decrementable = _CCCL_REQUIRES_EXPR((T), T i)((--i), (i--));

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view<int> io(0);
    auto iter1 = cuda::std::next(io.begin());
    auto iter2 = cuda::std::next(io.begin());
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert(cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }
  {
    cuda::std::ranges::iota_view io(SomeInt(0));
    auto iter1 = cuda::std::next(io.begin());
    auto iter2 = cuda::std::next(io.begin());
    assert(iter1 == iter2);
    assert(--iter1 != iter2--);
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_reference_v<decltype(iter2--)>);
    static_assert(cuda::std::is_reference_v<decltype(--iter2)>);
    static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(--iter2)>, decltype(iter2--)>);
  }

  static_assert(!Decrementable<cuda::std::ranges::iota_view<NotDecrementable>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iota_view::<iterator>::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/std/ranges>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    // Test `int`, which has operator<=>; the iota iterator should also have operator<=>.
    using R = cuda::std::ranges::iota_view<int>;
#if TEST_HAS_SPACESHIP()
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // TEST_HAS_SPACESHIP()

    decltype(auto) r = cuda::std::views::iota(42);
    static_assert(cuda::std::same_as<decltype(r), R>);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    assert(!(iter1 < iter1));
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(iter1 <= iter1);
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));
    assert(!(iter1 > iter2));
    assert(iter2 > iter1);
    assert(iter1 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter2 >= iter1);
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);
    assert(!(iter1 != iter1));
    assert(iter1 != iter2);
    assert(!(iter2 != iter2));

#if TEST_HAS_SPACESHIP()
    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
#endif // TEST_HAS_SPACESHIP()
  }

#if TEST_HAS_SPACESHIP()
  {
    // Test a new-school iterator with operator<=>; the iota iterator should also have operator<=>.
    using It = three_way_contiguous_iterator<int*>;
    static_assert(cuda::std::three_way_comparable<It>);
    using R = cuda::std::ranges::iota_view<It>;
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);

    int a[]                      = {1, 2, 3};
    cuda::std::same_as<R> auto r = cuda::std::views::iota(It(a));
    auto iter1                   = r.begin();
    auto iter2                   = iter1 + 1;

    assert(!(iter1 < iter1));
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(iter1 <= iter1);
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));
    assert(!(iter1 > iter2));
    assert(iter2 > iter1);
    assert(iter1 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter2 >= iter1);
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);
    assert(!(iter1 != iter1));
    assert(iter1 != iter2);
    assert(!(iter2 != iter2));

    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
  }
#endif // TEST_HAS_SPACESHIP()

  {
    // Test an old-school iterator with no operator<=>; the iota iterator shouldn't have operator<=> either.
    using It = random_access_iterator<int*>;
    using R  = cuda::std::ranges::iota_view<It>;
#if TEST_HAS_SPACESHIP()
    static_assert(!cuda::std::three_way_comparable<It>);
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // TEST_HAS_SPACESHIP()

    int a[]          = {1, 2, 3};
    decltype(auto) r = cuda::std::views::iota(It(a));
    static_assert(cuda::std::same_as<decltype(r), R>);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    assert(!(iter1 < iter1));
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(iter1 <= iter1);
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(!(iter1 > iter1));
    assert(!(iter1 > iter2));
    assert(iter2 > iter1);
    assert(iter1 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter2 >= iter1);
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);
    assert(!(iter1 != iter1));
    assert(iter1 != iter2);
    assert(!(iter2 != iter2));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// x += n;
// x + n;
// n + x;
// x -= n;
// x - n;
// x - y;
// All the arithmetic operators have the constraint `requires all-random-access<Const, Views...>;`,
// except `operator-(x, y)` which instead has the constraint
//    `requires (sized_sentinel_for<iterator_t<maybe-const<Const, Views>>,
//                                  iterator_t<maybe-const<Const, Views>>> && ...);`

#include <cuda/iterator>
#include <cuda/std/concepts>
#include <cuda/std/functional>

#include "test_macros.h"
#include "types.h"

template <class T, class U>
_CCCL_CONCEPT canPlusEqual = _CCCL_REQUIRES_EXPR((T, U), T& t, U& u)(t += u);

template <class T, class U>
_CCCL_CONCEPT canMinusEqual = _CCCL_REQUIRES_EXPR((T, U), T& t, U& u)(t -= u);

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4, 5};
  double b[] = {4.1, 3.2, 4.3, 0.1, 0.2};

  { // operator+(x, n) and operator+=
    cuda::zip_transform_iterator iter1{Plus{}, a, b};
    using Iter = decltype(iter1);

    const auto iter2 = iter1 + 2;
    assert(iter2 - iter1 == 2);
    static_assert(cuda::std::is_same_v<decltype(iter1 + 2), Iter>);
    const int expected1 = a[2] + static_cast<int>(b[2]);
    assert(*iter2 == expected1);

    const auto iter3 = 3 + iter1;
    assert(iter3 - iter1 == 3);
    static_assert(cuda::std::is_same_v<decltype(3 + iter1), Iter>);
    const int expected2 = a[3] + static_cast<int>(b[3]);
    assert(*iter3 == expected2);

    iter1 += 4;
    assert(iter1 - iter2 == 2);
    static_assert(cuda::std::is_same_v<decltype(iter1 += 4), Iter&>);
    const int expected3 = a[4] + static_cast<int>(b[4]);
    assert(*iter1 == expected3);
    static_assert(canPlusEqual<Iter, intptr_t>);
  }

  { // operator-(x, n) and operator-=
    cuda::zip_transform_iterator iter1{Plus{}, a + 5, b + 5};
    using Iter = decltype(iter1);

    const auto iter2 = iter1 - 3;
    assert(iter1 - iter2 == 3);
    static_assert(cuda::std::is_same_v<decltype(iter1 - 2), Iter>);
    const int expected1 = a[2] + static_cast<int>(b[2]);
    assert(*iter2 == expected1);

    iter1 -= 3;
    assert(iter2 == iter1);
    static_assert(cuda::std::is_same_v<decltype(iter1 -= 3), Iter&>);
    const int expected2 = a[2] + static_cast<int>(b[2]);
    assert(*iter1 == expected2);
    static_assert(canMinusEqual<Iter, intptr_t>);
  }

  { // operator-(x, y)
    cuda::zip_transform_iterator iter1{Plus{}, a, b};
    cuda::zip_transform_iterator iter2{Plus{}, a + 5, b + 5};
    using Iter = decltype(iter1);
    assert(iter2 - iter1 == 5);
    assert(iter1 - iter2 == -5);
    static_assert(cuda::std::is_same_v<decltype(iter2 - iter1), cuda::std::iter_difference_t<Iter>>);
  }

  { // One of the iterators is not random access but sized
    cuda::zip_transform_iterator iter1{Plus{}, forward_sized_iterator<>{a}, b};
    cuda::zip_transform_iterator iter2{Plus{}, forward_sized_iterator<>{a + 5}, b + 5};
    using Iter = decltype(iter1);
    assert(iter2 - iter1 == 5);
    assert(iter1 - iter2 == -5);

    static_assert(!cuda::std::invocable<cuda::std::plus<>, Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::plus<>, intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::minus<>, Iter, intptr_t>);
    static_assert(!canMinusEqual<Iter, intptr_t>);
  }

  { // One of the iterators is not random access and not sized
    cuda::zip_transform_iterator iter1{Plus{}, forward_iterator{a}, b};
    using Iter = decltype(iter1);
    static_assert(!cuda::std::invocable<cuda::std::plus<>, Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::plus<>, intptr_t, Iter>);
    static_assert(!canPlusEqual<Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::minus<>, Iter, intptr_t>);
    static_assert(!cuda::std::invocable<cuda::std::minus<>, Iter, Iter>);
    static_assert(!canMinusEqual<Iter, intptr_t>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}

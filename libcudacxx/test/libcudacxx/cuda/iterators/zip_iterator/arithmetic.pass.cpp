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
    cuda::zip_iterator iter1{a, b};
    using Iter = decltype(iter1);

    const auto iter2 = iter1 + 3;
    auto [x2, y2]    = *iter2;
    assert(cuda::std::addressof(x2) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(y2) == cuda::std::addressof(b[3]));
    static_assert(cuda::std::is_same_v<decltype(iter1 + 3), Iter>);

    const auto iter3 = 3 + iter1;
    auto [x3, y3]    = *iter3;
    assert(cuda::std::addressof(x3) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(y3) == cuda::std::addressof(b[3]));
    static_assert(cuda::std::is_same_v<decltype(3 + iter1), Iter>);

    iter1 += 3;
    assert(iter1 == iter2);
    auto [x1, y1] = *iter2;
    assert(cuda::std::addressof(x1) == cuda::std::addressof(a[3]));
    assert(cuda::std::addressof(y1) == cuda::std::addressof(b[3]));
    static_assert(cuda::std::is_same_v<decltype(iter1 += 3), Iter&>);

    static_assert(canPlusEqual<Iter, intptr_t>);
  }

  { // operator-(x, n) and operator-=
    cuda::zip_iterator iter1{a + 5, b + 5};
    using Iter = decltype(iter1);

    const auto iter2 = iter1 - 3;
    auto [x2, y2]    = *iter2;
    assert(cuda::std::addressof(x2) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(y2) == cuda::std::addressof(b[2]));
    static_assert(cuda::std::is_same_v<decltype(iter1 - 3), Iter>);

    iter1 -= 3;
    assert(iter1 == iter2);
    auto [x1, y1] = *iter2;
    assert(cuda::std::addressof(x1) == cuda::std::addressof(a[2]));
    assert(cuda::std::addressof(y1) == cuda::std::addressof(b[2]));
    static_assert(cuda::std::is_same_v<decltype(iter1 -= 3), Iter&>);

    static_assert(canMinusEqual<Iter, intptr_t>);
  }

  { // operator-(x, y)
    cuda::zip_iterator iter1{a, b};
    cuda::zip_iterator iter2{a + 5, b + 5};
    using Iter = decltype(iter1);
    assert(iter2 - iter1 == 5);
    assert(iter1 - iter2 == -5);
    static_assert(cuda::std::is_same_v<decltype(iter2 - iter1), cuda::std::iter_difference_t<Iter>>);
  }

  { // One of the iterators is not random access but sized
    cuda::zip_iterator iter1{forward_sized_iterator<>{a}, b};
    cuda::zip_iterator iter2{forward_sized_iterator<>{a + 5}, b + 5};
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
    cuda::zip_iterator iter1{forward_iterator{a}, b};
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

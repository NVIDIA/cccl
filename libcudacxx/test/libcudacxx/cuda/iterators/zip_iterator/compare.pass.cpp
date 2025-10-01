//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires (equality_comparable<iterator_t<maybe-const<Const, Views>>> && ...);
// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...>;
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires all-random-access<Const, Views...> &&
//            (three_way_comparable<iterator_t<maybe-const<Const, Views>>> && ...);

#include <cuda/iterator>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#include "test_iterators.h"
#include "types.h"

template <class Iter1, class Iter2>
__host__ __device__ constexpr void compareOperatorTest(Iter1&& iter1, Iter2&& iter2)
{
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

template <class Iter1, class Iter2>
__host__ __device__ constexpr void inequalityOperatorsDoNotExistTest(Iter1&& iter1, Iter2&& iter2)
{
  static_assert(!cuda::std::is_invocable_v<cuda::std::less<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::less_equal<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::greater<>, Iter1, Iter2>);
  static_assert(!cuda::std::is_invocable_v<cuda::std::greater_equal<>, Iter1, Iter2>);
}

__host__ __device__ constexpr bool test()
{
  int a[] = {1, 2, 3, 4};
  int b[] = {5, 6, 7, 8, 9};

#if TEST_HAS_SPACESHIP()
  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.

    using Iter = three_way_contiguous_iterator<int*>;
    static_assert(cuda::std::three_way_comparable<Iter>);
    static_assert(cuda::std::three_way_comparable<cuda::zip_iterator<Iter>>);

    cuda::zip_iterator iter1{Iter(a), Iter(b + 4)};
    cuda::zip_iterator iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
  }
#endif // TEST_HAS_SPACESHIP()

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have
    // operator<=> either.
    using Iter = random_access_iterator<int*>;
#if TEST_HAS_SPACESHIP()
    static_assert(!cuda::std::three_way_comparable<cuda::zip_iterator<Iter>>);
#endif // TEST_HAS_SPACESHIP()

    cuda::zip_iterator iter1{Iter(a), Iter(b + 4)};
    cuda::zip_iterator iter2{Iter(a + 1), Iter(b + 5)};

    compareOperatorTest(iter1, iter2);
  }

  {
    using Iter = cpp17_input_iterator<int*>;

    cuda::zip_iterator iter1{Iter(a), Iter(b + 4)};
    cuda::zip_iterator iter2{Iter(a + 1), Iter(b + 5)};

    assert(iter1 != iter2);
    ++iter1;
    assert(iter1 == iter2);

    inequalityOperatorsDoNotExistTest(iter1, iter2);
  }

  {
    // only < and == are needed
    using Iter = LessThanIterator;

    cuda::zip_iterator iter1{Iter(a), Iter(b + 4)};
    cuda::zip_iterator iter2{Iter(a + 1), Iter(b + 5)};
    static_assert(!cuda::std::invocable<cuda::std::equal_to<>, cuda::zip_iterator<Iter, Iter>>);
    compareOperatorTest(iter1, iter2);
  }

  {
    // underlying iterator does not support ==
    using Iter = cpp20_input_iterator<int*>;

    cuda::zip_iterator iter1{Iter(a), Iter(b + 4)};
    cuda::zip_iterator iter2{Iter(a + 1), Iter(b + 5)};
    static_assert(!cuda::std::invocable<cuda::std::equal_to<>, cuda::zip_iterator<Iter, Iter>>);
    inequalityOperatorsDoNotExistTest(iter1, iter2);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}

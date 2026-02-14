//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constant_iterator::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/iterator>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const cuda::strided_iterator<int*, Stride> iter1{buffer, stride};
    const cuda::strided_iterator<int*, Stride> iter2 = iter1 + 3;

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
    static_assert(cuda::std::three_way_comparable<Iter>);
    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
#endif // TEST_HAS_SPACESHIP()
  }

  {
    const cuda::strided_iterator<int*, Stride> iter1{buffer, stride};
    const cuda::strided_iterator<int*, int> iter2{buffer + 1, 5};

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
    static_assert(cuda::std::three_way_comparable<Iter>);
    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
#endif // TEST_HAS_SPACESHIP()
  }

  {
    static_assert(!cuda::std::equality_comparable_with<cuda::strided_iterator<int*>,
                                                       cuda::strided_iterator<random_access_iterator<int*>>>);
    static_assert(!cuda::std::totally_ordered_with<cuda::strided_iterator<int*>,
                                                   cuda::strided_iterator<random_access_iterator<int*>>>);
  }
}

__host__ __device__ constexpr bool test()
{
  test(2);
  test(Stride<2>{});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

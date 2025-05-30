//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_iterator::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/iterator>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
__host__ __device__ constexpr void test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  cuda::transform_iterator iter1{Iter{buffer}, PlusOne{}};
  cuda::transform_iterator iter2{Iter{buffer + 4}, PlusOne{}};

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
  // Test a new-school iterator with operator<=>; the transform iterator should also have operator<=>.
  if constexpr (cuda::std::is_same_v<Iter, three_way_contiguous_iterator<int*>>)
  {
    static_assert(cuda::std::three_way_comparable<Iter>);
    static_assert(cuda::std::three_way_comparable<decltype(iter1)>);

    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
  }
#endif // TEST_HAS_SPACESHIP()
}

__host__ __device__ constexpr bool test()
{
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

#if TEST_HAS_SPACESHIP()
  test<three_way_contiguous_iterator<int*>>();
#endif // TEST_HAS_SPACESHIP()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

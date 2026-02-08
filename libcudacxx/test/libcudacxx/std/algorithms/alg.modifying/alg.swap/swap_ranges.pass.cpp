//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   Iter2
//   swap_ranges(Iter1 first1, Iter1 last1, Iter2 first2);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class Iter2>
__host__ __device__ constexpr void test()
{
  using iter_value_t = typename cuda::std::remove_reference<decltype(*cuda::std::declval<Iter1>())>::type;

  {
    iter_value_t i[3] = {1, 2, 3};
    iter_value_t j[3] = {4, 5, 6};
    Iter2 r           = cuda::std::swap_ranges(Iter1(i), Iter1(i + 3), Iter2(j));
    assert(base(r) == j + 3);
    assert(i[0] == 4);
    assert(i[1] == 5);
    assert(i[2] == 6);
    assert(j[0] == 1);
    assert(j[1] == 2);
    assert(j[2] == 3);
  }

  {
    iter_value_t src[2][2]  = {{0, 1}, {2, 3}};
    iter_value_t dest[2][2] = {{9, 8}, {7, 6}};

    cuda::std::swap(src, dest);

    assert(src[0][0] == 9);
    assert(src[0][1] == 8);
    assert(src[1][0] == 7);
    assert(src[1][1] == 6);

    assert(dest[0][0] == 0);
    assert(dest[0][1] == 1);
    assert(dest[1][0] == 2);
    assert(dest[1][1] == 3);
  }
}

template <class T>
__host__ __device__ constexpr bool test()
{
  test<forward_iterator<T*>, forward_iterator<T*>>();
  test<forward_iterator<T*>, bidirectional_iterator<T*>>();
  test<forward_iterator<T*>, random_access_iterator<T*>>();
  test<forward_iterator<T*>, T*>();

  test<bidirectional_iterator<T*>, forward_iterator<T*>>();
  test<bidirectional_iterator<T*>, bidirectional_iterator<T*>>();
  test<bidirectional_iterator<T*>, random_access_iterator<T*>>();
  test<bidirectional_iterator<T*>, T*>();

  test<random_access_iterator<T*>, forward_iterator<T*>>();
  test<random_access_iterator<T*>, bidirectional_iterator<T*>>();
  test<random_access_iterator<T*>, random_access_iterator<T*>>();
  test<random_access_iterator<T*>, T*>();

  test<T*, forward_iterator<T*>>();
  test<T*, bidirectional_iterator<T*>>();
  test<T*, random_access_iterator<T*>>();
  test<T*, T*>();

  return true;
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<MoveOnly>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

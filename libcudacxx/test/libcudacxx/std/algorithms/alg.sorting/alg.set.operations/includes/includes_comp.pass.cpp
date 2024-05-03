//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2, typename Compare>
//   requires Predicate<Compare, Iter1::value_type, Iter2::value_type>
//         && Predicate<Compare, Iter2::value_type, Iter1::value_type>
//   constexpr bool             // constexpr after C++17
//   includes(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class Iter2>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]          = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ib[]          = {4, 2};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  int ic[]          = {2, 1};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  ((void) sc);
  int id[]          = {3, 3, 3, 3};
  const unsigned sd = sizeof(id) / sizeof(id[0]);
  ((void) sd);

  assert(cuda::std::includes(Iter1(ia), Iter1(ia), Iter2(ib), Iter2(ib), cuda::std::greater<int>()));
  assert(!cuda::std::includes(Iter1(ia), Iter1(ia), Iter2(ib), Iter2(ib + 1), cuda::std::greater<int>()));
  assert(cuda::std::includes(Iter1(ia), Iter1(ia + 1), Iter2(ib), Iter2(ib), cuda::std::greater<int>()));
  assert(cuda::std::includes(Iter1(ia), Iter1(ia + sa), Iter2(ia), Iter2(ia + sa), cuda::std::greater<int>()));

  assert(cuda::std::includes(Iter1(ia), Iter1(ia + sa), Iter2(ib), Iter2(ib + sb), cuda::std::greater<int>()));
  assert(!cuda::std::includes(Iter1(ib), Iter1(ib + sb), Iter2(ia), Iter2(ia + sa), cuda::std::greater<int>()));

  assert(cuda::std::includes(Iter1(ia + 8), Iter1(ia + sa), Iter2(ic), Iter2(ic + sc), cuda::std::greater<int>()));
  assert(!cuda::std::includes(Iter1(ia + 8), Iter1(ia + sa), Iter2(ib), Iter2(ib + sb), cuda::std::greater<int>()));

  assert(cuda::std::includes(Iter1(ia), Iter1(ia + sa), Iter2(id), Iter2(id + 1), cuda::std::greater<int>()));
  assert(cuda::std::includes(Iter1(ia), Iter1(ia + sa), Iter2(id), Iter2(id + 2), cuda::std::greater<int>()));
  assert(cuda::std::includes(Iter1(ia), Iter1(ia + sa), Iter2(id), Iter2(id + 3), cuda::std::greater<int>()));
  assert(!cuda::std::includes(Iter1(ia), Iter1(ia + sa), Iter2(id), Iter2(id + 4), cuda::std::greater<int>()));
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, const int*>();

  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>>();
  test<forward_iterator<const int*>, const int*>();

  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, const int*>();

  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>>();
  test<random_access_iterator<const int*>, const int*>();

  test<const int*, cpp17_input_iterator<const int*>>();
  test<const int*, forward_iterator<const int*>>();
  test<const int*, bidirectional_iterator<const int*>>();
  test<const int*, random_access_iterator<const int*>>();
  test<const int*, const int*>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}

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

// template<ForwardIterator Iter>
//   requires OutputIterator<Iter, Iter::reference>
//         && EqualityComparable<Iter::value_type>
//   constexpr Iter        // constexpr after C++17
//   unique(Iter first, Iter last);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ constexpr void test()
{
  using iter_value_t = typename cuda::std::remove_reference<decltype(*cuda::std::declval<Iter>())>::type;
  iter_value_t ia[]  = {0};
  const unsigned sa  = sizeof(ia) / sizeof(ia[0]);
  Iter r             = cuda::std::unique(Iter(ia), Iter(ia + sa));
  assert(base(r) == ia + sa);
  assert(ia[0] == 0);

  iter_value_t ib[] = {0, 1};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  r                 = cuda::std::unique(Iter(ib), Iter(ib + sb));
  assert(base(r) == ib + sb);
  assert(ib[0] == 0);
  assert(ib[1] == 1);

  iter_value_t ic[] = {0, 0};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  r                 = cuda::std::unique(Iter(ic), Iter(ic + sc));
  assert(base(r) == ic + 1);
  assert(ic[0] == 0);

  iter_value_t id[] = {0, 0, 1};
  const unsigned sd = sizeof(id) / sizeof(id[0]);
  r                 = cuda::std::unique(Iter(id), Iter(id + sd));
  assert(base(r) == id + 2);
  assert(id[0] == 0);
  assert(id[1] == 1);

  iter_value_t ie[] = {0, 0, 1, 0};
  const unsigned se = sizeof(ie) / sizeof(ie[0]);
  r                 = cuda::std::unique(Iter(ie), Iter(ie + se));
  assert(base(r) == ie + 3);
  assert(ie[0] == 0);
  assert(ie[1] == 1);
  assert(ie[2] == 0);

  iter_value_t ig[] = {0, 0, 1, 1};
  const unsigned sg = sizeof(ig) / sizeof(ig[0]);
  r                 = cuda::std::unique(Iter(ig), Iter(ig + sg));
  assert(base(r) == ig + 2);
  assert(ig[0] == 0);
  assert(ig[1] == 1);

  iter_value_t ih[] = {0, 1, 1};
  const unsigned sh = sizeof(ih) / sizeof(ih[0]);
  r                 = cuda::std::unique(Iter(ih), Iter(ih + sh));
  assert(base(r) == ih + 2);
  assert(ih[0] == 0);
  assert(ih[1] == 1);

  iter_value_t ii[] = {0, 1, 1, 1, 2, 2, 2};
  const unsigned si = sizeof(ii) / sizeof(ii[0]);
  r                 = cuda::std::unique(Iter(ii), Iter(ii + si));
  assert(base(r) == ii + 3);
  assert(ii[0] == 0);
  assert(ii[1] == 1);
  assert(ii[2] == 2);
}

__host__ __device__ constexpr bool test()
{
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  test<forward_iterator<MoveOnly*>>();
  test<bidirectional_iterator<MoveOnly*>>();
  test<random_access_iterator<MoveOnly*>>();
  test<MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

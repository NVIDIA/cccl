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

// template<ForwardIterator Iter, EquivalenceRelation<auto, Iter::value_type> Pred>
//   requires OutputIterator<Iter, RvalueOf<Iter::reference>::type>
//         && CopyConstructible<Pred>
//   constexpr Iter        // constexpr after C++17
//   unique(Iter first, Iter last, Pred pred);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

struct count_equal
{
  __host__ __device__ constexpr count_equal(int& count) noexcept
      : count_(count)
  {}
  int& count_;
  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator()(const T& x, const T& y) const noexcept
  {
    ++count_;
    return x == y;
  }
};

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  using iter_value_t = typename cuda::std::remove_reference<decltype(*cuda::std::declval<Iter>())>::type;

  int count_equal_count = 0;
  count_equal count_op{count_equal_count};

  iter_value_t ia[] = {0};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  count_equal_count = 0;
  Iter r            = cuda::std::unique(Iter(ia), Iter(ia + sa), count_op);
  assert(base(r) == ia + sa);
  assert(ia[0] == 0);
  assert(count_equal_count == sa - 1);

  iter_value_t ib[] = {0, 1};
  const unsigned sb = sizeof(ib) / sizeof(ib[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(ib), Iter(ib + sb), count_op);
  assert(base(r) == ib + sb);
  assert(ib[0] == 0);
  assert(ib[1] == 1);
  assert(count_equal_count == sb - 1);

  iter_value_t ic[] = {0, 0};
  const unsigned sc = sizeof(ic) / sizeof(ic[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(ic), Iter(ic + sc), count_op);
  assert(base(r) == ic + 1);
  assert(ic[0] == 0);
  assert(count_equal_count == sc - 1);

  iter_value_t id[] = {0, 0, 1};
  const unsigned sd = sizeof(id) / sizeof(id[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(id), Iter(id + sd), count_op);
  assert(base(r) == id + 2);
  assert(id[0] == 0);
  assert(id[1] == 1);
  assert(count_equal_count == sd - 1);

  iter_value_t ie[] = {0, 0, 1, 0};
  const unsigned se = sizeof(ie) / sizeof(ie[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(ie), Iter(ie + se), count_op);
  assert(base(r) == ie + 3);
  assert(ie[0] == 0);
  assert(ie[1] == 1);
  assert(ie[2] == 0);
  assert(count_equal_count == se - 1);

  iter_value_t ig[] = {0, 0, 1, 1};
  const unsigned sg = sizeof(ig) / sizeof(ig[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(ig), Iter(ig + sg), count_op);
  assert(base(r) == ig + 2);
  assert(ig[0] == 0);
  assert(ig[1] == 1);
  assert(count_equal_count == sg - 1);

  iter_value_t ih[] = {0, 1, 1};
  const unsigned sh = sizeof(ih) / sizeof(ih[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(ih), Iter(ih + sh), count_op);
  assert(base(r) == ih + 2);
  assert(ih[0] == 0);
  assert(ih[1] == 1);
  assert(count_equal_count == sh - 1);

  iter_value_t ii[] = {0, 1, 1, 1, 2, 2, 2};
  const unsigned si = sizeof(ii) / sizeof(ii[0]);
  count_equal_count = 0;
  r                 = cuda::std::unique(Iter(ii), Iter(ii + si), count_op);
  assert(base(r) == ii + 3);
  assert(ii[0] == 0);
  assert(ii[1] == 1);
  assert(ii[2] == 2);
  assert(count_equal_count == si - 1);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
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
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}

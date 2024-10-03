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

// template<ShuffleIterator Iter>
//   Iter
//   rotate(Iter first, Iter middle, Iter last);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  using iter_value_t = typename cuda::std::remove_reference<decltype(*cuda::std::declval<Iter>())>::type;

  iter_value_t ia[] = {0};
  const int sa      = static_cast<int>(sizeof(ia) / sizeof(ia[0]));
  Iter r            = cuda::std::rotate(Iter(ia), Iter(ia), Iter(ia));
  assert(base(r) == ia);
  assert(ia[0] == 0);
  r = cuda::std::rotate(Iter(ia), Iter(ia), Iter(ia + sa));
  assert(base(r) == ia + sa);
  assert(ia[0] == 0);
  r = cuda::std::rotate(Iter(ia), Iter(ia + sa), Iter(ia + sa));
  assert(base(r) == ia);
  assert(ia[0] == 0);

  iter_value_t ib[] = {0, 1};
  const int sb      = static_cast<int>(sizeof(ib) / sizeof(ib[0]));
  r                 = cuda::std::rotate(Iter(ib), Iter(ib), Iter(ib + sb));
  assert(base(r) == ib + sb);
  assert(ib[0] == 0);
  assert(ib[1] == 1);
  r = cuda::std::rotate(Iter(ib), Iter(ib + 1), Iter(ib + sb));
  assert(base(r) == ib + 1);
  assert(ib[0] == 1);
  assert(ib[1] == 0);
  r = cuda::std::rotate(Iter(ib), Iter(ib + sb), Iter(ib + sb));
  assert(base(r) == ib);
  assert(ib[0] == 1);
  assert(ib[1] == 0);

  iter_value_t ic[] = {0, 1, 2};
  const int sc      = static_cast<int>(sizeof(ic) / sizeof(ic[0]));
  r                 = cuda::std::rotate(Iter(ic), Iter(ic), Iter(ic + sc));
  assert(base(r) == ic + sc);
  assert(ic[0] == 0);
  assert(ic[1] == 1);
  assert(ic[2] == 2);
  r = cuda::std::rotate(Iter(ic), Iter(ic + 1), Iter(ic + sc));
  assert(base(r) == ic + 2);
  assert(ic[0] == 1);
  assert(ic[1] == 2);
  assert(ic[2] == 0);
  r = cuda::std::rotate(Iter(ic), Iter(ic + 2), Iter(ic + sc));
  assert(base(r) == ic + 1);
  assert(ic[0] == 0);
  assert(ic[1] == 1);
  assert(ic[2] == 2);
  r = cuda::std::rotate(Iter(ic), Iter(ic + sc), Iter(ic + sc));
  assert(base(r) == ic);
  assert(ic[0] == 0);
  assert(ic[1] == 1);
  assert(ic[2] == 2);

  iter_value_t id[] = {0, 1, 2, 3};
  const int sd      = static_cast<int>(sizeof(id) / sizeof(id[0]));
  r                 = cuda::std::rotate(Iter(id), Iter(id), Iter(id + sd));
  assert(base(r) == id + sd);
  assert(id[0] == 0);
  assert(id[1] == 1);
  assert(id[2] == 2);
  assert(id[3] == 3);
  r = cuda::std::rotate(Iter(id), Iter(id + 1), Iter(id + sd));
  assert(base(r) == id + 3);
  assert(id[0] == 1);
  assert(id[1] == 2);
  assert(id[2] == 3);
  assert(id[3] == 0);
  r = cuda::std::rotate(Iter(id), Iter(id + 2), Iter(id + sd));
  assert(base(r) == id + 2);
  assert(id[0] == 3);
  assert(id[1] == 0);
  assert(id[2] == 1);
  assert(id[3] == 2);
  r = cuda::std::rotate(Iter(id), Iter(id + 3), Iter(id + sd));
  assert(base(r) == id + 1);
  assert(id[0] == 2);
  assert(id[1] == 3);
  assert(id[2] == 0);
  assert(id[3] == 1);
  r = cuda::std::rotate(Iter(id), Iter(id + sd), Iter(id + sd));
  assert(base(r) == id);
  assert(id[0] == 2);
  assert(id[1] == 3);
  assert(id[2] == 0);
  assert(id[3] == 1);

  iter_value_t ie[] = {0, 1, 2, 3, 4};
  const int se      = static_cast<int>(sizeof(ie) / sizeof(ie[0]));
  r                 = cuda::std::rotate(Iter(ie), Iter(ie), Iter(ie + se));
  assert(base(r) == ie + se);
  assert(ie[0] == 0);
  assert(ie[1] == 1);
  assert(ie[2] == 2);
  assert(ie[3] == 3);
  assert(ie[4] == 4);
  r = cuda::std::rotate(Iter(ie), Iter(ie + 1), Iter(ie + se));
  assert(base(r) == ie + 4);
  assert(ie[0] == 1);
  assert(ie[1] == 2);
  assert(ie[2] == 3);
  assert(ie[3] == 4);
  assert(ie[4] == 0);
  r = cuda::std::rotate(Iter(ie), Iter(ie + 2), Iter(ie + se));
  assert(base(r) == ie + 3);
  assert(ie[0] == 3);
  assert(ie[1] == 4);
  assert(ie[2] == 0);
  assert(ie[3] == 1);
  assert(ie[4] == 2);
  r = cuda::std::rotate(Iter(ie), Iter(ie + 3), Iter(ie + se));
  assert(base(r) == ie + 2);
  assert(ie[0] == 1);
  assert(ie[1] == 2);
  assert(ie[2] == 3);
  assert(ie[3] == 4);
  assert(ie[4] == 0);
  r = cuda::std::rotate(Iter(ie), Iter(ie + 4), Iter(ie + se));
  assert(base(r) == ie + 1);
  assert(ie[0] == 0);
  assert(ie[1] == 1);
  assert(ie[2] == 2);
  assert(ie[3] == 3);
  assert(ie[4] == 4);
  r = cuda::std::rotate(Iter(ie), Iter(ie + se), Iter(ie + se));
  assert(base(r) == ie);
  assert(ie[0] == 0);
  assert(ie[1] == 1);
  assert(ie[2] == 2);
  assert(ie[3] == 3);
  assert(ie[4] == 4);

  iter_value_t ig[] = {0, 1, 2, 3, 4, 5};
  const int sg      = static_cast<int>(sizeof(ig) / sizeof(ig[0]));
  r                 = cuda::std::rotate(Iter(ig), Iter(ig), Iter(ig + sg));
  assert(base(r) == ig + sg);
  assert(ig[0] == 0);
  assert(ig[1] == 1);
  assert(ig[2] == 2);
  assert(ig[3] == 3);
  assert(ig[4] == 4);
  assert(ig[5] == 5);
  r = cuda::std::rotate(Iter(ig), Iter(ig + 1), Iter(ig + sg));
  assert(base(r) == ig + 5);
  assert(ig[0] == 1);
  assert(ig[1] == 2);
  assert(ig[2] == 3);
  assert(ig[3] == 4);
  assert(ig[4] == 5);
  assert(ig[5] == 0);
  r = cuda::std::rotate(Iter(ig), Iter(ig + 2), Iter(ig + sg));
  assert(base(r) == ig + 4);
  assert(ig[0] == 3);
  assert(ig[1] == 4);
  assert(ig[2] == 5);
  assert(ig[3] == 0);
  assert(ig[4] == 1);
  assert(ig[5] == 2);
  r = cuda::std::rotate(Iter(ig), Iter(ig + 3), Iter(ig + sg));
  assert(base(r) == ig + 3);
  assert(ig[0] == 0);
  assert(ig[1] == 1);
  assert(ig[2] == 2);
  assert(ig[3] == 3);
  assert(ig[4] == 4);
  assert(ig[5] == 5);
  r = cuda::std::rotate(Iter(ig), Iter(ig + 4), Iter(ig + sg));
  assert(base(r) == ig + 2);
  assert(ig[0] == 4);
  assert(ig[1] == 5);
  assert(ig[2] == 0);
  assert(ig[3] == 1);
  assert(ig[4] == 2);
  assert(ig[5] == 3);
  r = cuda::std::rotate(Iter(ig), Iter(ig + 5), Iter(ig + sg));
  assert(base(r) == ig + 1);
  assert(ig[0] == 3);
  assert(ig[1] == 4);
  assert(ig[2] == 5);
  assert(ig[3] == 0);
  assert(ig[4] == 1);
  assert(ig[5] == 2);
  r = cuda::std::rotate(Iter(ig), Iter(ig + sg), Iter(ig + sg));
  assert(base(r) == ig);
  assert(ig[0] == 3);
  assert(ig[1] == 4);
  assert(ig[2] == 5);
  assert(ig[3] == 0);
  assert(ig[4] == 1);
  assert(ig[5] == 2);
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
#if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}

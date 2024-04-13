//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <class U>
// reverse_iterator(const reverse_iterator<U> &u); // constexpr since C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

struct Base
{};
struct Derived : Base
{};

template <class It, class U>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(U u)
{
  const cuda::std::reverse_iterator<U> r2(u);
  cuda::std::reverse_iterator<It> r1 = r2;
  assert(base(r1.base()) == base(u));
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  Derived d{};
  test<bidirectional_iterator<Base*>>(bidirectional_iterator<Derived*>(&d));
  test<random_access_iterator<const Base*>>(random_access_iterator<Derived*>(&d));
  test<Base*>(&d);
  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER > 2011
  static_assert(tests(), "");
#endif
  return 0;
}

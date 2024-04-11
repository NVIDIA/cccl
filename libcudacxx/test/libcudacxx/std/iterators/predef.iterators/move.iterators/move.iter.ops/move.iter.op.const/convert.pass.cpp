//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_iterator

// template <class U>
//   requires HasConstructor<Iter, const U&>
//   move_iterator(const move_iterator<U> &u);
//
//  constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It, class U>
__host__ __device__ void test(U u)
{
  const cuda::std::move_iterator<U> r2(u);
  cuda::std::move_iterator<It> r1 = r2;
  assert(base(r1.base()) == base(u));
}

struct Base
{};
struct Derived : Base
{};

int main(int, char**)
{
  Derived d;

  test<cpp17_input_iterator<Base*>>(cpp17_input_iterator<Derived*>(&d));
  test<forward_iterator<Base*>>(forward_iterator<Derived*>(&d));
  test<bidirectional_iterator<Base*>>(bidirectional_iterator<Derived*>(&d));
  test<random_access_iterator<const Base*>>(random_access_iterator<Derived*>(&d));
  test<Base*>(&d);

#if TEST_STD_VER > 2011
  {
    constexpr const Derived* p                             = nullptr;
    constexpr cuda::std::move_iterator<const Derived*> it1 = cuda::std::make_move_iterator(p);
    constexpr cuda::std::move_iterator<const Base*> it2(it1);
    static_assert(it2.base() == p, "");
  }
#endif

  return 0;
}

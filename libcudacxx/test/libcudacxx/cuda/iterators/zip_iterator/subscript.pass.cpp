//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto operator[](difference_type n) const requires
//        all_random_access<Const, Views...>

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT canSubscript = _CCCL_REQUIRES_EXPR((Iter), Iter it)(it[0]);

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3};

  { // single iterator
    cuda::zip_iterator iter{random_access_iterator{a}};
    assert(cuda::std::addressof(cuda::std::get<0>(iter[0])) == cuda::std::addressof(a[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), cuda::std::tuple<int&>>);
  }

  { // single iterator, operator* is const
    const cuda::zip_iterator iter{a};
    assert(cuda::std::addressof(cuda::std::get<0>(iter[0])) == cuda::std::addressof(a[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), cuda::std::tuple<int&>>);
  }

  { // two different iterators
    cuda::zip_iterator iter{a, b};
    auto [x, y] = iter[0];
    assert(cuda::std::addressof(x) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(y) == cuda::std::addressof(b[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), cuda::std::pair<int&, double&>>);

    x = 5;
    y = 0.1;
    assert(a[0] == 5);
    assert(b[0] == 0.1);
  }

  { // iterator that generates prvalues
    cuda::zip_iterator iter{a, b, cuda::counting_iterator{0}};
    assert(cuda::std::addressof(cuda::std::get<0>(iter[0])) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(cuda::std::get<1>(iter[0])) == cuda::std::addressof(b[0]));
    assert(cuda::std::get<2>(iter[0]) == 0);
    static_assert(cuda::std::is_same_v<decltype(iter[0]), cuda::std::tuple<int&, double&, int>>);
  }

  { // const-correctness
    cuda::zip_iterator iter{a, cuda::std::as_const(b)};
    assert(cuda::std::addressof(cuda::std::get<0>(iter[0])) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(cuda::std::get<1>(iter[0])) == cuda::std::addressof(b[0]));
    static_assert(cuda::std::is_same_v<decltype(iter[0]), cuda::std::pair<int&, const double&>>);
  }

  { // not all random_access_iterator
    [[maybe_unused]] cuda::zip_iterator iter{a, forward_iterator{b}, cuda::counting_iterator{0}};
    static_assert(!canSubscript<decltype(iter)>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto operator*() const;

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3};

  { // single iterator
    cuda::zip_iterator iter{a};
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[0]));
    static_assert(cuda::std::is_same_v<decltype(*iter), cuda::std::tuple<int&>>);
  }

  { // single iterator, operator* is const
    const cuda::zip_iterator iter{a};
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[0]));
    static_assert(cuda::std::is_same_v<decltype(*iter), cuda::std::tuple<int&>>);
  }

  { // two different iterators
    cuda::zip_iterator iter{a, b};
    auto [x, y] = *iter;
    assert(cuda::std::addressof(x) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(y) == cuda::std::addressof(b[0]));
    static_assert(cuda::std::is_same_v<decltype(*iter), cuda::std::tuple<int&, double&>>);

    x = 5;
    y = 0.1;
    assert(a[0] == 5);
    assert(b[0] == 0.1);
  }

  { // iterator that generates prvalues
    cuda::zip_iterator iter{a, b, cuda::counting_iterator{0}};
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[0]));
    assert(cuda::std::get<2>(*iter) == 0);
    static_assert(cuda::std::is_same_v<decltype(*iter), cuda::std::tuple<int&, double&, int>>);
  }

  { // const-correctness
    cuda::zip_iterator iter{a, cuda::std::as_const(b)};
    assert(cuda::std::addressof(cuda::std::get<0>(*iter)) == cuda::std::addressof(a[0]));
    assert(cuda::std::addressof(cuda::std::get<1>(*iter)) == cuda::std::addressof(b[0]));
    static_assert(cuda::std::is_same_v<decltype(*iter), cuda::std::tuple<int&, const double&>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}

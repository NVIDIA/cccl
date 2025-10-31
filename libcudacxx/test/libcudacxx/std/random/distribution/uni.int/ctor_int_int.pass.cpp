//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// explicit uniform_int_distribution(IntType a = 0,
//                                   IntType b = numeric_limits<IntType>::max()); // before C++20
// uniform_int_distribution() : uniform_int_distribution(0) {}                    // C++20
// explicit uniform_int_distribution(IntType a,
//                                   IntType b = numeric_limits<IntType>::max()); // C++20

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "make_implicit.h"
#include "test_convertible.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_implicit()
{
  using D = cuda::std::uniform_int_distribution<>;
  static_assert(test_convertible<D>(), "");
  assert(D(0) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
  static_assert(!test_convertible<D, T, T>(), "");
}

int main(int, char**)
{
  {
    using D = cuda::std::uniform_int_distribution<>;
    D d;
    assert(d.a() == 0);
    assert(d.b() == cuda::std::numeric_limits<int>::max());
  }
  {
    using D = cuda::std::uniform_int_distribution<>;
    D d(-6);
    assert(d.a() == -6);
    assert(d.b() == cuda::std::numeric_limits<int>::max());
  }
  {
    using D = cuda::std::uniform_int_distribution<>;
    D d(-6, 106);
    assert(d.a() == -6);
    assert(d.b() == 106);
  }

  test_implicit<int>();
  test_implicit<long>();

  return 0;
}

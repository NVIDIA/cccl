//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class normal_distribution

// explicit normal_distribution(RealType mean = 0.0, RealType stddev = 1.0); // before C++20
// normal_distribution() : normal_distribution(0.0) {}                       // C++20
// explicit normal_distribution(RealType mean, RealType stddev = 1.0);       // C++20

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "make_implicit.h"
#include "test_convertible.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_implicit()
{
  using D = cuda::std::normal_distribution<T>;
  static_assert(test_convertible<D>(), "");
  assert(D(0) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
  static_assert(!test_convertible<D, T, T>(), "");
}

int main(int, char**)
{
  {
    using D = cuda::std::normal_distribution<>;
    D d;
    assert(d.mean() == 0);
    assert(d.stddev() == 1);
  }
  {
    using D = cuda::std::normal_distribution<>;
    D d(14.5);
    assert(d.mean() == 14.5);
    assert(d.stddev() == 1);
  }
  {
    using D = cuda::std::normal_distribution<>;
    D d(14.5, 5.25);
    assert(d.mean() == 14.5);
    assert(d.stddev() == 5.25);
  }

  test_implicit<float>();
  test_implicit<double>();

  return 0;
}

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
// class uniform_real_distribution

// template<class _URng> result_type operator()(_URng& g, const param_type& parm);

#include <cuda/std/__memory_>
#include <cuda/std/__random_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/numeric>
#include <cuda/std/span>

#include "test_macros.h"

template <class T>
__host__ __device__ inline T sqr(T x)
{
  return x * x;
}

__host__ __device__ void test()
{
  using D = cuda::std::uniform_real_distribution<>;
  using G = cuda::std::minstd_rand;
  using P = D::param_type;
  G g;
  D d(5.5, 100);
  P p(-10, 20);
  constexpr int N                               = 10000;
  cuda::std::unique_ptr<D::result_type[]> array = cuda::std::make_unique<D::result_type[]>(N);
  cuda::std::span<D::result_type, N> u{array.get(), array.get() + N};
  for (int i = 0; i < N; ++i)
  {
    D::result_type v = d(g, p);
    assert(p.a() <= v && v < p.b());
    u[i] = v;
  }
  D::result_type mean     = cuda::std::accumulate(u.begin(), u.end(), D::result_type(0)) / u.size();
  D::result_type var      = 0;
  D::result_type skew     = 0;
  D::result_type kurtosis = 0;
  for (cuda::std::size_t i = 0; i < u.size(); ++i)
  {
    D::result_type dbl = (u[i] - mean);
    D::result_type d2  = sqr(dbl);
    var += d2;
    skew += dbl * d2;
    kurtosis += d2 * d2;
  }
  var /= u.size();
  D::result_type dev = cuda::std::sqrt(var);
  skew /= u.size() * dev * var;
  kurtosis /= u.size() * var * var;
  kurtosis -= 3;
  D::result_type x_mean     = (p.a() + p.b()) / 2;
  D::result_type x_var      = sqr(p.b() - p.a()) / 12;
  D::result_type x_skew     = 0;
  D::result_type x_kurtosis = -6. / 5;
  assert(cuda::std::abs((mean - x_mean) / x_mean) < 0.1);
  assert(cuda::std::abs((var - x_var) / x_var) < 0.1);
  assert(cuda::std::abs(skew - x_skew) < 0.01);
  assert(cuda::std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.1);
}

int main(int, char**)
{
  test();
  return 0;
}

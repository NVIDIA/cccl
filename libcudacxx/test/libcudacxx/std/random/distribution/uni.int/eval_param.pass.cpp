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

int main(int, char**)
{
  {
    using D = cuda::std::uniform_int_distribution<>;
    using G = cuda::std::minstd_rand0;
    using P = D::param_type;
    G g;
    D d(5, 100);
    P p(-10, 20);
    constexpr int N                               = 10000;
    cuda::std::unique_ptr<D::result_type[]> array = cuda::std::make_unique<D::result_type[]>(N);
    cuda::std::span<D::result_type, N> u{array.get(), array.get() + N};
    for (int i = 0; i < N; ++i)
    {
      D::result_type v = d(g, p);
      assert(p.a() <= v && v <= p.b());
      u[i] = v;
    }
    double mean     = cuda::std::accumulate(u.begin(), u.end(), double(0)) / u.size();
    double var      = 0;
    double skew     = 0;
    double kurtosis = 0;
    for (cuda::std::size_t i = 0; i < u.size(); ++i)
    {
      double dbl = (u[i] - mean);
      double d2  = sqr(dbl);
      var += d2;
      skew += dbl * d2;
      kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = cuda::std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean     = ((double) p.a() + p.b()) / 2;
    double x_var      = (sqr((double) p.b() - p.a() + 1) - 1) / 12;
    double x_skew     = 0;
    double x_kurtosis = -6. * (sqr((double) p.b() - p.a() + 1) + 1) / (5. * (sqr((double) p.b() - p.a() + 1) - 1));
    assert(cuda::std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(cuda::std::abs((var - x_var) / x_var) < 0.01);
    assert(cuda::std::abs(skew - x_skew) < 0.02);
    assert(cuda::std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.01);
  }

  return 0;
}

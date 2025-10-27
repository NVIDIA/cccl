//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <cuda/std/__memory_>
#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/numeric>
#include <cuda/std/span>

#include "test_macros.h"

template <class T>
inline __host__ __device__ T sqr(T x)
{
  return x * x;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    typedef cuda::std::bernoulli_distribution D;
    typedef cuda::std::minstd_rand G;
    G g;
    D d(.75);
    const int N                                   = 100000;
    cuda::std::unique_ptr<D::result_type[]> array = cuda::std::make_unique<D::result_type[]>(N);
    cuda::std::span<D::result_type, N> u{array.get(), array.get() + N};
    for (int i = 0; i < N; ++i)
    {
      u[i] = d(g);
    }
    double mean     = cuda::std::accumulate(u.begin(), u.end(), double(0)) / u.size();
    double var      = 0;
    double skew     = 0;
    double kurtosis = 0;
    for (std::size_t i = 0; i < u.size(); ++i)
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
    double x_mean     = d.p();
    double x_var      = d.p() * (1 - d.p());
    double x_skew     = (1 - 2 * d.p()) / cuda::std::sqrt(x_var);
    double x_kurtosis = (6 * sqr(d.p()) - 6 * d.p() + 1) / x_var;
    assert(cuda::std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(cuda::std::abs((var - x_var) / x_var) < 0.01);
    assert(cuda::std::abs((skew - x_skew) / x_skew) < 0.02);
    assert(cuda::std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.05);
  }
  {
    typedef cuda::std::bernoulli_distribution D;
    typedef cuda::std::minstd_rand G;
    G g;
    D d(.25);
    const int N                                   = 100000;
    cuda::std::unique_ptr<D::result_type[]> array = cuda::std::make_unique<D::result_type[]>(N);
    cuda::std::span<D::result_type, N> u{array.get(), array.get() + N};
    for (int i = 0; i < N; ++i)
    {
      u[i] = d(g);
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
    double x_mean     = d.p();
    double x_var      = d.p() * (1 - d.p());
    double x_skew     = (1 - 2 * d.p()) / cuda::std::sqrt(x_var);
    double x_kurtosis = (6 * sqr(d.p()) - 6 * d.p() + 1) / x_var;
    assert(cuda::std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(cuda::std::abs((var - x_var) / x_var) < 0.01);
    assert(cuda::std::abs((skew - x_skew) / x_skew) < 0.02);
    assert(cuda::std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.05);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif
  return 0;
}

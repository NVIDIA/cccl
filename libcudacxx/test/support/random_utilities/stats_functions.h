//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCUDACXX_TEST_SUPPORT_STATS_FUNCTIONS_H
#define LIBCUDACXX_TEST_SUPPORT_STATS_FUNCTIONS_H

#include <cuda/std/cmath>

// Regularized incomplete gamma function P(a,x)
// Adapted from numerical recipes
__host__ __device__ inline double incomplete_gamma(double a, double x)
{
  if (x <= 0.0)
  {
    return 0.0;
  }

  const int max_iter = 100;
  double sum         = 1.0 / a;
  double term        = 1.0 / a;

  for (int n = 1; n < max_iter; ++n)
  {
    term *= x / (a + n);
    sum += term;
    if (cuda::std::abs(term) < 1e-12 * cuda::std::abs(sum))
    {
      break;
    }
  }

  return cuda::std::exp(-x + a * cuda::std::log(x) - cuda::std::lgamma(a)) * sum;
}

// Regularized incomplete beta function I_x(a,b)
// Adapted from numerical recipes
__host__ __device__ inline double incomplete_beta(double a, double b, double x)
{
  if (x <= 0.0)
  {
    return 0.0;
  }
  if (x >= 1.0)
  {
    return 1.0;
  }
  double log_beta = cuda::std::lgamma(a) + cuda::std::lgamma(b) - cuda::std::lgamma(a + b);

  const int max_iter = 200;
  const double eps   = 1e-12;

  bool use_complement = false;
  double xx           = x;
  double aa           = a;
  double bb           = b;

  if (x > (a + 1.0) / (a + b + 2.0))
  {
    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    use_complement = true;
    xx             = 1.0 - x;
    aa             = b;
    bb             = a;
  }

  // Continued fraction expansion
  double qab = aa + bb;
  double qap = aa + 1.0;
  double qam = aa - 1.0;

  double c = 1.0;
  double d = 1.0 - qab * xx / qap;

  if (cuda::std::abs(d) < 1e-30)
  {
    d = 1e-30;
  }
  d        = 1.0 / d;
  double h = d;

  for (int m = 1; m <= max_iter; ++m)
  {
    int m2     = 2 * m;
    double aa1 = m * (bb - m) * xx / ((qam + m2) * (aa + m2));
    d          = 1.0 + aa1 * d;
    if (cuda::std::abs(d) < 1e-30)
    {
      d = 1e-30;
    }
    c = 1.0 + aa1 / c;
    if (cuda::std::abs(c) < 1e-30)
    {
      c = 1e-30;
    }
    d = 1.0 / d;
    h *= d * c;

    double aa2 = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2));
    d          = 1.0 + aa2 * d;
    if (cuda::std::abs(d) < 1e-30)
    {
      d = 1e-30;
    }
    c = 1.0 + aa2 / c;
    if (cuda::std::abs(c) < 1e-30)
    {
      c = 1e-30;
    }
    d          = 1.0 / d;
    double del = d * c;
    h *= del;

    if (cuda::std::abs(del - 1.0) < eps)
    {
      break;
    }
  }

  double log_prefix = aa * cuda::std::log(xx) + bb * cuda::std::log(1.0 - xx) - log_beta;
  double result     = cuda::std::exp(log_prefix) * h / aa;

  if (use_complement)
  {
    return 1.0 - result;
  }
  return result;
}

#endif // LIBCUDACXX_TEST_SUPPORT_STATS_FUNCTIONS_H

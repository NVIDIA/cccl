// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 end-to-end transcendental math via the Black-Scholes formula.
//
//  pi.pass.cpp chains the arithmetic; this one chains the math functions. A price
//  runs exp, log, sqrt and erfc into each other and then subtracts two terms of
//  similar size, so a function that is accurate in isolation but mis-normalizes
//  its result shows up here in a way the one-operation-at-a-time tests cannot see.
//
//  Two independent checks per case. The first compares against the same formula in
//  native double. The second is put-call parity,
//
//      C - P == S*exp(-qT) - K*exp(-rT),
//
//  an exact identity of the model: it needs no reference at all, and it constrains
//  the two prices jointly, so a consistent bias in erfc that both prices share
//  cancels in the first check but not in this one.
//
//  Accuracy expected of the math functions is binary64, not double-double. On any
//  target without native fp128 -- every GPU before Blackwell, and any host without
//  libquadmath -- fp64mp2 math widens through double, so requiring more would make
//  the bound depend on the build rather than on the library. fp32mp2 has dedicated
//  double-float implementations and is held to its own epsilon; higher accuracy
//  from fp64mp2 where the hardware offers it is a bonus this test does not demand.
//  Arithmetic is not covered by that relaxation: pi.pass.cpp holds add, sub, mul,
//  div, fma and sqrt to the full precision of the type.
//
//  Measured worst case is 2.6x the tolerance base, identical on host and device and
//  in both fp128 configurations, against a bound of 16x. Prices are normalized by
//  the spot rather than by themselves, because a deep out-of-the-money price is
//  near zero, where a relative error carries no information.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/fpmp_math>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// The native-double control resolves to these; the fpmp2 types find their own by
// ADL. Non-template overloads so double picks them exactly.
TEST_HOST_DEVICE_FUNC double mp_sqrt(double __x)
{
  return ::cuda::std::sqrt(__x);
}
TEST_HOST_DEVICE_FUNC double mp_exp(double __x)
{
  return ::cuda::std::exp(__x);
}
TEST_HOST_DEVICE_FUNC double mp_log(double __x)
{
  return ::cuda::std::log(__x);
}
TEST_HOST_DEVICE_FUNC double mp_erfc(double __x)
{
  return ::cuda::std::erfc(__x);
}
template <class T>
TEST_HOST_DEVICE_FUNC T mp_sqrt(const T& __x)
{
  return sqrt(__x);
}
template <class T>
TEST_HOST_DEVICE_FUNC T mp_exp(const T& __x)
{
  return exp(__x);
}
template <class T>
TEST_HOST_DEVICE_FUNC T mp_log(const T& __x)
{
  return log(__x);
}
template <class T>
TEST_HOST_DEVICE_FUNC T mp_erfc(const T& __x)
{
  return erfc(__x);
}

struct bs_case
{
  double spot, strike, rate, div, vol, maturity;
};

// The allowed error as a multiple of the weaker of T's own precision and binary64,
// which is the accuracy the math functions are held to (see the header comment).
template <class T>
TEST_HOST_DEVICE_FUNC double tol_base()
{
  const double __eps_t = (double) ::cuda::std::numeric_limits<T>::epsilon();
  const double __eps_d = (double) ::cuda::std::numeric_limits<double>::epsilon();
  return __eps_t > __eps_d ? __eps_t : __eps_d;
}

// N(x) = erfc(-x/sqrt(2)) / 2, with 1/sqrt(2) built at the working precision of T
// instead of from a binary64 literal, so the argument reduction is not the limit.
template <class T>
TEST_HOST_DEVICE_FUNC T norm_cdf(const T& __x, const T& __inv_sqrt2, const T& __half)
{
  return __half * mp_erfc(-__x * __inv_sqrt2);
}

//   d1 = (log(S/K) + (r - q + vol^2/2)*T) / (vol*sqrt(T)),  d2 = d1 - vol*sqrt(T)
//   C  = S*exp(-qT)*N(d1) - K*exp(-rT)*N(d2)
//   P  = K*exp(-rT)*N(-d2) - S*exp(-qT)*N(-d1)
//
// `one` is the caller's volatile-loaded 1.0. Every input is scaled by it -- exactly,
// so the values are unchanged -- which keeps the case table from becoming a constant
// expression the host compiler could fold, or the device pass a precomputed answer.
template <class T>
TEST_HOST_DEVICE_FUNC void bs_call_put(const bs_case& __c, double __one, T* __call, T* __put, T* __s_disc, T* __k_disc)
{
  const T __two{__one + __one};
  const T __half      = T{__one} / __two;
  const T __inv_sqrt2 = T{__one} / mp_sqrt(__two);

  const T __s{__c.spot * __one};
  const T __k{__c.strike * __one};
  const T __r{__c.rate * __one};
  const T __q{__c.div * __one};
  const T __vol{__c.vol * __one};
  const T __t{__c.maturity * __one};

  const T __vsqrt_t = __vol * mp_sqrt(__t);
  const T __d1      = (mp_log(__s / __k) + (__r - __q + __half * __vol * __vol) * __t) / __vsqrt_t;
  const T __d2      = __d1 - __vsqrt_t;

  *__s_disc = __s * mp_exp(-__q * __t);
  *__k_disc = __k * mp_exp(-__r * __t);

  *__call = *__s_disc * norm_cdf(__d1, __inv_sqrt2, __half) - *__k_disc * norm_cdf(__d2, __inv_sqrt2, __half);
  *__put  = *__k_disc * norm_cdf(-__d2, __inv_sqrt2, __half) - *__s_disc * norm_cdf(-__d1, __inv_sqrt2, __half);
}

// tol_mult is the allowed error as a multiple of tol_base<T>().
template <class T>
TEST_HOST_DEVICE_FUNC void check_case(const bs_case& __c, double __one, double __tol_mult)
{
  T __call, __put, __s_disc, __k_disc;
  bs_call_put<T>(__c, __one, &__call, &__put, &__s_disc, &__k_disc);

  double __call_d, __put_d, __s_disc_d, __k_disc_d;
  bs_call_put<double>(__c, __one, &__call_d, &__put_d, &__s_disc_d, &__k_disc_d);

  // Absolute tolerance on the scale of the problem, which is the spot.
  const double __tol = __tol_mult * tol_base<T>() * __c.spot;

  assert(::cuda::std::fabs((double) __call - __call_d) <= __tol);
  assert(::cuda::std::fabs((double) __put - __put_d) <= __tol);

  const T __parity = (__call - __put) - (__s_disc - __k_disc);
  assert(::cuda::std::fabs((double) __parity) <= __tol);

  // No-arbitrage bounds, exact inequalities in the model and independent of the
  // reference: a call is worth at least its discounted intrinsic value and never
  // more than the discounted spot, and neither price is negative.
  const double __intrinsic = (double) __s_disc - (double) __k_disc;
  assert((double) __call >= (__intrinsic > 0.0 ? __intrinsic : 0.0) - __tol);
  assert((double) __call <= (double) __s_disc + __tol);
  assert((double) __put >= -__tol);
}

template <class T>
TEST_HOST_DEVICE_FUNC void check_all(const bs_case* __cases, int __n, double __one, double __tol_mult)
{
  for (int __i = 0; __i < __n; ++__i)
  {
    check_case<T>(__cases[__i], __one, __tol_mult);
  }
}

TEST_HOST_DEVICE_FUNC void test()
{
  // See bs_call_put: the seed keeps the case table opaque to constant folding, so
  // the math under test really runs on whichever target the test runs on.
  volatile double one_v = 1.0;
  const double one      = one_v;

  // At the money, in and out of the money, high volatility, near expiry, and one
  // deep out of the money case where the call price is ~1e-13 of the spot and the
  // two terms of the difference agree to most of their digits.
  const bs_case cases[] = {
    {100.0, 100.0, 0.05, 0.02, 0.20, 1.00},
    {100.0, 80.0, 0.05, 0.00, 0.20, 1.00},
    {100.0, 130.0, 0.05, 0.00, 0.20, 1.00},
    {100.0, 100.0, 0.03, 0.01, 0.80, 2.00},
    {100.0, 100.0, 0.05, 0.00, 0.15, 0.02},
    {50.0, 200.0, 0.04, 0.01, 0.25, 0.50},
  };
  const int n = (int) (sizeof(cases) / sizeof(cases[0]));

  // Worst measured is 2.6, so this leaves a factor of 6 against target-to-target
  // variation in the underlying libm while still catching a real break, which
  // misses by orders of magnitude rather than by a few multiples.
  constexpr double tol_mult = 16.0;

  check_all<double>(cases, n, one, tol_mult); // control

  // fpmp2_accuracy::def aliases mid, so the def and mid rows are the same type.
  check_all<cudax::fp32mp2>(cases, n, one, tol_mult);
  check_all<cudax::fp32mp2_high>(cases, n, one, tol_mult);
  check_all<cudax::fp32mp2_mid>(cases, n, one, tol_mult);
  check_all<cudax::fp32mp2_low>(cases, n, one, tol_mult);

  check_all<cudax::fp64mp2>(cases, n, one, tol_mult);
  check_all<cudax::fp64mp2_high>(cases, n, one, tol_mult);
  check_all<cudax::fp64mp2_mid>(cases, n, one, tol_mult);
  check_all<cudax::fp64mp2_low>(cases, n, one, tol_mult);
}

int main(int, char**)
{
  test();

  return 0;
}

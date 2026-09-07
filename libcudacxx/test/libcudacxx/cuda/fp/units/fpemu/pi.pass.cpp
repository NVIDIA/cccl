// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu end-to-end arithmetic via a Gauss-Legendre Pi computation.
//
//  The other unit tests check one operation at a time against a reference. This
//  one runs a real algorithm that chains all of them -- add, sub, mul, div, sqrt,
//  fma, the compound assignments, comparison and unary minus -- inside a single
//  kernel, and asks whether Pi comes out. Gauss-Legendre suits the purpose
//  because it needs no transcendental function (fpemu supplies none) and
//  converges quadratically, so four iterations exhaust binary64 precision and any
//  error in the chain shows up in the digits rather than being absorbed.
//
//  Each accuracy tag is run for both the packed and the unpacked representation
//  and compared against Pi rounded to binary64, with native double as the control.
//  The tolerances come from measurement: every mode but `low` lands within a few
//  ulp of binary64, while `low` settles around 2e-7 because its add and multiply
//  keep only about half of the binary64 mantissa. The seed comes through a
//  volatile so the whole computation happens at run time on whichever target the
//  test is running on.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Pi rounded to binary64. The Gauss-Legendre result is compared to this, so the
// reference costs no precision the test could measure.
constexpr double kPi = 3.14159265358979323846;

// The native-double control resolves to these; the fpemu types find their own
// sqrt / fma by ADL. Non-template overloads so double picks them exactly.
TEST_HOST_DEVICE_FUNC double emu_sqrt(double __x)
{
  return ::cuda::std::sqrt(__x);
}
TEST_HOST_DEVICE_FUNC double emu_fma(double __x, double __y, double __z)
{
  return ::cuda::std::fma(__x, __y, __z);
}
template <class T>
TEST_HOST_DEVICE_FUNC T emu_sqrt(const T& __x)
{
  return sqrt(__x);
}
template <class T>
TEST_HOST_DEVICE_FUNC T emu_fma(const T& __x, const T& __y, const T& __z)
{
  return fma(__x, __y, __z);
}

// Gauss-Legendre:
//   a0 = 1, b0 = 1/sqrt(2), t0 = 1/4, p0 = 1
//   a' = (a + b)/2,  b' = sqrt(a*b),  t' = t - p*(a - a')^2,  p' = 2*p
//   Pi ~ (a + b)^2 / (4*t)
// The number of correct digits doubles per iteration, so the iteration count is
// the whole precision budget of the test.
//
// `one` is the caller's volatile-loaded 1.0, and every operand here is derived
// from it, so nothing in this function is a constant expression.
template <class T>
TEST_HOST_DEVICE_FUNC double gauss_legendre_pi(double one, int iters)
{
  const T two{one + one};
  const T four{two * two};

  T a{one};
  T b = T{one} / emu_sqrt(two);
  T t = T{one} / four;
  T p{one};

  // |a - b| contracts quadratically. Tracking it exercises comparison and unary
  // minus, and pins the contraction the algorithm depends on: a step that fails
  // to renormalize shows up as a gap that stops shrinking.
  T gap = a;
  gap -= b;

  for (int i = 0; i < iters; ++i)
  {
    T a_next = a + b;
    a_next /= two;

    const T d = a - a_next;
    b         = emu_sqrt(a * b);
    t         = emu_fma(-p, d * d, t); // t -= p*d*d
    a         = a_next;
    p *= two;

    T gap_next = a;
    gap_next -= b;
    if (gap_next < T{one - one})
    {
      gap_next = -gap_next;
    }
    assert(!(gap_next > gap));
    gap = gap_next;
  }

  T sum = a;
  sum += b;

  T den = four;
  den *= t;
  return (double) ((sum * sum) / den);
}

template <class T>
TEST_HOST_DEVICE_FUNC void check_pi(double one, int iters, double tol)
{
  const double got = gauss_legendre_pi<T>(one, iters);
  assert(::cuda::std::fabs((got - kPi) / kPi) <= tol);
}

TEST_FUNC void test()
{
  // Every input below is derived from this seed, and a literal seed would make
  // the whole computation a constant expression: the host compiler would then be
  // free to fold it, and the device pass would load a precomputed answer instead
  // of executing fpemu. Reading the seed through a volatile makes it opaque, so
  // the arithmetic under test really runs on whichever target the test runs on.
  volatile double one_v = 1.0;
  const double one      = one_v;

  // Four iterations is where the algorithm saturates binary64; measured relative
  // error is then <= 6e-16 for every mode except `low`, which sits at 2e-7
  // because its add and multiply truncate the mantissa to about 24 bits.
  constexpr int iters      = 4;
  constexpr double tol     = 1.0e-15;
  constexpr double tol_low = 1.0e-6;

  check_pi<double>(one, iters, tol); // control

  check_pi<cudax::fp64emu>(one, iters, tol);
  check_pi<cudax::fp64emu_high>(one, iters, tol);
  check_pi<cudax::fp64emu_mid>(one, iters, tol);
  check_pi<cudax::fp64emu_low>(one, iters, tol_low);

  check_pi<cudax::fp64emu_unpacked>(one, iters, tol);
  check_pi<cudax::fp64emu_unpacked_high>(one, iters, tol);
  check_pi<cudax::fp64emu_unpacked_mid>(one, iters, tol);
  check_pi<cudax::fp64emu_unpacked_low>(one, iters, tol_low);
}

int main(int, char**)
{
  test();

  return 0;
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 end-to-end arithmetic via a Gauss-Legendre Pi computation.
//
//  The other unit tests check one operation at a time against a reference. This
//  one runs a real algorithm that chains all of them -- add, sub, mul, div, sqrt,
//  fma, the compound assignments, comparison and unary minus -- inside a single
//  kernel, and asks whether Pi comes out. Gauss-Legendre suits the purpose
//  because it needs no transcendental function and converges quadratically, so
//  four iterations exhaust even the double-double precision of fp64mp2 and any
//  error in the chain shows up in the digits rather than being absorbed.
//
//  Every accuracy tag of both precisions is run, with native double as the
//  control. The result is compared against Pi carried at the working precision of
//  the type under test, not against binary64: fp64mp2 holds about 106 bits, so a
//  binary64 reference would cap the measurable error at 1e-16 and hide whether
//  the extra precision is really delivered. Measured relative error at four
//  iterations is 1.3 eps for double, <= 0.7 eps for the fp32mp2 tags and
//  <= 0.3 eps for the fp64mp2 tags, so one rule -- a small multiple of the type's
//  own epsilon -- serves every type, and fp64mp2 is held to 1e-30 rather than to
//  1e-16. The seed comes through a volatile so the whole computation happens at
//  run time on whichever target the test is running on.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Pi rounded to binary64, and the residual Pi - (double) Pi. The pair carries Pi
// to 1e-33, i.e. 0.01 eps of the double-double types, so the reference itself
// contributes nothing measurable to the comparison.
constexpr double kPi   = 3.14159265358979323846;
constexpr double kPiLo = 1.2246467991473531772e-16;

// The native-double control resolves to these; the fpmp2 types find their own
// sqrt / fma by ADL. Non-template overloads so double picks them exactly.
TEST_HOST_DEVICE_FUNC double mp_sqrt(double __x)
{
  return ::cuda::std::sqrt(__x);
}
TEST_HOST_DEVICE_FUNC double mp_fma(double __x, double __y, double __z)
{
  return ::cuda::std::fma(__x, __y, __z);
}
template <class T>
TEST_HOST_DEVICE_FUNC T mp_sqrt(const T& __x)
{
  return sqrt(__x);
}
template <class T>
TEST_HOST_DEVICE_FUNC T mp_fma(const T& __x, const T& __y, const T& __z)
{
  return fma(__x, __y, __z);
}

// Pi at the working precision of T, so the comparison never costs precision the
// type could have shown.
template <class T>
TEST_HOST_DEVICE_FUNC T pi_ref()
{
  if constexpr (::cuda::std::is_same_v<decltype(T().hi()), float>)
  {
    // A float pair resolves about 45 bits, so splitting the binary64 constant
    // leaves a reference carrying 0.04 eps of the type -- far below what the
    // comparison can see.
    return T(kPi);
  }
  else
  {
    // A double pair resolves about 103 bits, which binary64 alone cannot fill:
    // hi + lo is the canonical double-double Pi.
    return T(kPi, kPiLo);
  }
}
template <>
TEST_HOST_DEVICE_FUNC double pi_ref<double>()
{
  return kPi;
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
TEST_HOST_DEVICE_FUNC T gauss_legendre_pi(double one, int iters)
{
  const T two{one + one};
  const T four{two * two};
  const T zero{one - one};

  T a{one};
  T b = T{one} / mp_sqrt(two);
  T t = T{one} / four;
  T p{one};

  // |a - b| contracts quadratically. Tracking it exercises comparison and unary
  // minus, and pins the contraction the algorithm depends on: a step that fails
  // to renormalize shows up as a gap that stops shrinking.
  T gap = a;
  gap -= b;

  // Contraction is only required while the gap still carries information. Once it
  // reaches the noise floor the remaining bits are rounding debris, and a tag that
  // renormalizes lazily (`low`) can jitter there by a fraction of an eps, so the
  // check would otherwise depend on the iteration count rather than on precision.
  const T noise_floor{::cuda::std::numeric_limits<T>::epsilon()};

  for (int i = 0; i < iters; ++i)
  {
    T a_next = a + b;
    a_next /= two;

    const T d = a - a_next;
    b         = mp_sqrt(a * b);
    t         = mp_fma(-p, d * d, t); // t -= p*d*d
    a         = a_next;
    p *= two;

    T gap_next = a;
    gap_next -= b;
    if (gap_next < zero)
    {
      gap_next = -gap_next;
    }
    if (gap > noise_floor)
    {
      assert(!(gap_next > gap));
    }
    gap = gap_next;
  }

  T sum = a;
  sum += b;

  T den = four;
  den *= t;
  return (sum * sum) / den;
}

// tol_eps is the allowed error as a multiple of the type's own epsilon.
template <class T>
TEST_HOST_DEVICE_FUNC void check_pi(double one, int iters, double tol_eps)
{
  const T got      = gauss_legendre_pi<T>(one, iters);
  const T residual = got - pi_ref<T>();
  const double eps = (double) ::cuda::std::numeric_limits<T>::epsilon();
  assert(::cuda::std::fabs((double) residual / kPi) <= tol_eps * eps);
}

TEST_HOST_DEVICE_FUNC void test()
{
  // Every input below is derived from this seed, and a literal seed would make
  // the whole computation a constant expression: the host compiler would then be
  // free to fold it, and the device pass would load a precomputed answer instead
  // of executing fpmp. Reading the seed through a volatile makes it opaque, so
  // the arithmetic under test really runs on whichever target the test runs on.
  volatile double one_v = 1.0;
  const double one      = one_v;

  // Four iterations is where the algorithm saturates the widest type here
  // (fp64mp2, about 106 bits); binary64 and the fp32mp2 tags saturate at three.
  // The bound is a multiple of each type's epsilon, so it tracks the precision
  // rather than a hard-coded decimal: the largest measured error is 1.3 eps, and
  // a break in the chain misses by orders of magnitude rather than by a few eps.
  constexpr int iters      = 4;
  constexpr double tol_eps = 16.0;

  check_pi<double>(one, iters, tol_eps); // control

  // fpmp2_accuracy::def aliases mid, so the def and mid rows are the same type.
  check_pi<cudax::fp32mp2>(one, iters, tol_eps);
  check_pi<cudax::fp32mp2_high>(one, iters, tol_eps);
  check_pi<cudax::fp32mp2_mid>(one, iters, tol_eps);
  check_pi<cudax::fp32mp2_low>(one, iters, tol_eps);

  check_pi<cudax::fp64mp2>(one, iters, tol_eps);
  check_pi<cudax::fp64mp2_high>(one, iters, tol_eps);
  check_pi<cudax::fp64mp2_mid>(one, iters, tol_eps);
  check_pi<cudax::fp64mp2_low>(one, iters, tol_eps);
}

int main(int, char**)
{
  test();

  return 0;
}

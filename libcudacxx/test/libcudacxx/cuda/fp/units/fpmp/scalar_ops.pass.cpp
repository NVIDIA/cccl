// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: arithmetic between a pair and a scalar that one limb holds exactly.
//
//  At every accuracy level: x + c and x * c within 2u^2 of the exact result,
//  x + c and x / c equal to the pair operation with c promoted, x * y within
//  its first-order bound (6u^2 at mid, 5u^2 otherwise), and the compound forms
//  and integer scalars equal to the operators.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

// a + b = s + e exactly.
template <typename T>
TEST_HOST_DEVICE_FUNC T two_sum(T a, T b, T& e)
{
  T s  = a + b;
  T ap = s - b;
  T bp = s - ap;
  e    = (a - ap) + (b - bp);
  return s;
}

// a * b = p + e exactly.
template <typename T>
TEST_HOST_DEVICE_FUNC T two_prod(T a, T b, T& e)
{
  T p = a * b;
  e   = cuda::std::fma(a, b, -p);
  return p;
}

// An exact sum of up to 16 terms, kept as a nonoverlapping expansion.
template <typename T>
struct expansion
{
  T comp[16];
  int n = 0;

  TEST_HOST_DEVICE_FUNC void add(T t)
  {
    int m = 0;
    for (int i = 0; i < n; ++i)
    {
      T e;
      t = two_sum(t, comp[i], e);
      if (e != T(0))
      {
        comp[m++] = e;
      }
    }
    if (t != T(0))
    {
      comp[m++] = t;
    }
    n = m;
  }
  // Components are in increasing magnitude, so this is accurate to a few units of the
  // last place of the result.
  TEST_HOST_DEVICE_FUNC T value() const
  {
    T s = T(0);
    for (int i = 0; i < n; ++i)
    {
      s += comp[i];
    }
    return s;
  }
};

template <typename T>
TEST_HOST_DEVICE_FUNC T unit_roundoff()
{
  return cuda::std::ldexp(T(1), -cuda::std::numeric_limits<T>::digits);
}

// Deterministic generator of normalized pairs and scalars.
template <typename T>
struct generator
{
  cuda::std::uint64_t state;

  TEST_HOST_DEVICE_FUNC double uniform()
  {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * 0x1.0p-53;
  }
  TEST_HOST_DEVICE_FUNC T sign()
  {
    return uniform() < 0.5 ? T(-1) : T(1);
  }
  // Significand in [1, 2); regime 2 concentrates near the binade boundaries.
  TEST_HOST_DEVICE_FUNC T significand(int regime)
  {
    if (regime == 2)
    {
      double d = cuda::std::ldexp(uniform(), -static_cast<int>(uniform() * cuda::std::numeric_limits<T>::digits) - 1);
      T m      = static_cast<T>(uniform() < 0.5 ? 1.0 + d : 2.0 - d);
      return m < T(2) ? m : T(1);
    }
    T m = static_cast<T>(1.0 + uniform());
    return m < T(2) ? m : T(1);
  }
  TEST_HOST_DEVICE_FUNC T scalar(int regime)
  {
    return sign() * cuda::std::ldexp(significand(regime), static_cast<int>(uniform() * 9) - 4);
  }
  // Regime 0 draws the low limb uniformly; regimes 1 and 2 near its largest magnitude.
  TEST_HOST_DEVICE_FUNC void pair(int regime, T& hi, T& lo)
  {
    int e = static_cast<int>(uniform() * 9) - 4;
    hi    = sign() * cuda::std::ldexp(significand(regime), e);
    T r   = static_cast<T>(regime == 0 ? uniform() : 1.0 - cuda::std::ldexp(uniform(), -8));
    lo    = sign() * cuda::std::ldexp(r, e - cuda::std::numeric_limits<T>::digits);
    T s   = hi + lo;
    lo    = lo - (s - hi);
    hi    = s;
  }
};

template <typename T>
TEST_HOST_DEVICE_FUNC T relative_error(const expansion<T>& error, const expansion<T>& exact)
{
  return cuda::std::fabs(error.value()) / cuda::std::fabs(exact.value());
}

// x * c within 2u^2 of the exact product.
template <typename P, typename T>
TEST_HOST_DEVICE_FUNC void check_scalar_product(const P& x, T c)
{
  const T u2  = unit_roundoff<T>() * unit_roundoff<T>();
  const T tol = T(1) + T(64) * unit_roundoff<T>();
  const P r   = x * c;
  expansion<T> exact, error;
  T e1, e2;
  const T p1 = two_prod(x.hi(), c, e1);
  const T p2 = two_prod(x.lo(), c, e2);
  exact.add(p1);
  exact.add(e1);
  exact.add(p2);
  exact.add(e2);
  error.add(r.hi());
  error.add(r.lo());
  error.add(-p1);
  error.add(-e1);
  error.add(-p2);
  error.add(-e2);
  assert(relative_error(error, exact) <= T(2) * u2 * tol);
}

template <typename P>
TEST_HOST_DEVICE_FUNC void test_accuracy()
{
  using T     = decltype(P{}.hi());
  const T u2  = unit_roundoff<T>() * unit_roundoff<T>();
  const T tol = T(1) + T(64) * unit_roundoff<T>();
  // Operands on which promoting c to the pair errs by 2.9u^2.
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    check_scalar_product(P(0x1.03e8fcp+0f, -0x1.ffc7b8p-25f), 0x1.02f33ap+0f);
  }
  else
  {
    check_scalar_product(P(0x1.007a6d0ffa6e7p+0, -0x1.fee1ab634ap-54), 0x1.059a003e98835p+0);
  }
  generator<T> gen{12345};
  for (int regime = 0; regime < 3; ++regime)
  {
    for (int i = 0; i < 512; ++i)
    {
      T xh, xl, yh, yl;
      gen.pair(regime, xh, xl);
      gen.pair(regime, yh, yl);
      const T c = gen.scalar(regime);
      const P x(xh, xl);
      const P y(yh, yl);

      // x + c: within 2u^2 of the exact sum, and equal to the promoted pair addition.
      {
        const P r = x + c;
        expansion<T> exact, error;
        exact.add(xh);
        exact.add(xl);
        exact.add(c);
        error.add(r.hi());
        error.add(r.lo());
        error.add(-xh);
        error.add(-xl);
        error.add(-c);
        if (exact.n == 0)
        {
          assert(error.n == 0);
        }
        else
        {
          assert(relative_error(error, exact) <= T(2) * u2 * tol);
        }
        const P promoted = x + P(c);
        assert(r.hi() == promoted.hi() && r.lo() == promoted.lo());
      }
      // x - c and c - x: within 2u^2 of the exact difference, computed independently.
      for (int order = 0; order < 2; ++order)
      {
        const P r    = order == 0 ? x - c : c - x;
        const T sign = order == 0 ? T(1) : T(-1);
        expansion<T> exact, error;
        exact.add(sign * xh);
        exact.add(sign * xl);
        exact.add(-sign * c);
        error.add(r.hi());
        error.add(r.lo());
        error.add(-sign * xh);
        error.add(-sign * xl);
        error.add(sign * c);
        if (exact.n == 0)
        {
          assert(error.n == 0);
        }
        else
        {
          assert(relative_error(error, exact) <= T(2) * u2 * tol);
        }
      }
      check_scalar_product(x, c);
      // x / c: the promoted pair division without the term on the divisor's zero low limb.
      {
        const P r        = x / c;
        const P promoted = x / P(c);
        assert(r.hi() == promoted.hi() && r.lo() == promoted.lo());
      }
      // x * y: within its first-order bound, 6u^2 at mid and 5u^2 at low and high.
      {
        const P r = x * y;
        expansion<T> exact, error;
        T e[4];
        T p[4] = {two_prod(xh, yh, e[0]), two_prod(xh, yl, e[1]), two_prod(xl, yh, e[2]), two_prod(xl, yl, e[3])};
        error.add(r.hi());
        error.add(r.lo());
        for (int k = 0; k < 4; ++k)
        {
          exact.add(p[k]);
          exact.add(e[k]);
          error.add(-p[k]);
          error.add(-e[k]);
        }
        const T bound = cuda::std::is_same_v<P, cuda::fpmp2<T, cuda::fpmp2_accuracy::mid>> ? T(6) : T(5);
        assert(relative_error(error, exact) <= bound * u2 * tol);
      }
    }
  }
}

// Scalar-first subtraction on fixed operands with results computed by hand.
template <typename P>
TEST_HOST_DEVICE_FUNC void test_scalar_minus_pair()
{
  using T   = decltype(P{}.hi());
  const T u = unit_roundoff<T>();
  // 3 - (1 + u/2) = 2 - u/2: the high limb rounds to 2, and the low limb is exactly -u/2.
  {
    const P r = T(3) - P(T(1), u / T(2));
    expansion<T> error;
    error.add(r.hi());
    error.add(r.lo());
    error.add(T(-2));
    error.add(u / T(2));
    assert(error.n == 0);
    if (!cuda::std::is_same_v<P, cuda::fpmp2<T, cuda::fpmp2_accuracy::low>>)
    {
      assert(r.hi() == T(2) && r.lo() == -u / T(2));
    }
  }
  // A low limb of 4u overlaps a high limb of 1 (half its ulp is u): 3 - (1 + 4u) = 2 - 4u,
  // which one limb holds.
  {
    const P r = T(3) - P(T(1), T(4) * u);
    expansion<T> error;
    error.add(r.hi());
    error.add(r.lo());
    error.add(T(-2));
    error.add(T(4) * u);
    assert(error.n == 0);
  }
}

// The compound forms, both operand orders and the integer scalars reach the same kernels.
template <typename P>
TEST_HOST_DEVICE_FUNC void test_forms()
{
  using T = decltype(P{}.hi());
  generator<T> gen{777};
  for (int i = 0; i < 64; ++i)
  {
    T xh, xl;
    gen.pair(i % 3, xh, xl);
    const P x(xh, xl);
    const T c = gen.scalar(i % 3);

    const P sum = x + c;
    const P dif = x - c;
    const P rdf = c - x;
    const P prd = x * c;
    const P quo = x / c;
    assert((c + x).hi() == sum.hi() && (c + x).lo() == sum.lo());
    assert((c * x).hi() == prd.hi() && (c * x).lo() == prd.lo());
    assert(dif.hi() == (x + (-c)).hi() && dif.lo() == (x + (-c)).lo());
    assert(rdf.hi() == ((-x) + c).hi() && rdf.lo() == ((-x) + c).lo());

    P a = x;
    a += c;
    assert(a.hi() == sum.hi() && a.lo() == sum.lo());
    a = x;
    a -= c;
    assert(a.hi() == dif.hi() && a.lo() == dif.lo());
    a = x;
    a *= c;
    assert(a.hi() == prd.hi() && a.lo() == prd.lo());
    a = x;
    a /= c;
    assert(a.hi() == quo.hi() && a.lo() == quo.lo());

    // A 16-bit integer fits one limb of either element type.
    const cuda::std::int16_t k = static_cast<cuda::std::int16_t>(i - 32);
    assert((x * k).hi() == (x * static_cast<T>(k)).hi() && (x * k).lo() == (x * static_cast<T>(k)).lo());
    assert((x + k).hi() == (x + static_cast<T>(k)).hi() && (x + k).lo() == (x + static_cast<T>(k)).lo());
    a = x;
    a *= k;
    assert(a.hi() == (x * static_cast<T>(k)).hi() && a.lo() == (x * static_cast<T>(k)).lo());
  }

  // Increments go through the single-limb addition.
  P a(T(1.5), T(0));
  ++a;
  assert(a.hi() == T(2.5) && a.lo() == T(0));
  a--;
  assert(a.hi() == T(1.5) && a.lo() == T(0));
}

// The accuracy-selecting functions take the same kernels with a scalar operand.
template <typename P>
TEST_HOST_DEVICE_FUNC void test_accuracy_selecting()
{
  using T = decltype(P{}.hi());
  generator<T> gen{4242};
  T xh, xl;
  gen.pair(0, xh, xl);
  const P x(xh, xl);
  const T c = gen.scalar(0);

  const P m = cuda::mul<cuda::fpmp2_accuracy::high>(x, c);
  const P n = cuda::mul<cuda::fpmp2_accuracy::high>(c, x);
  assert(m.hi() == n.hi() && m.lo() == n.lo());
  const P s = cuda::sub<cuda::fpmp2_accuracy::low>(c, x);
  const P t = cuda::add<cuda::fpmp2_accuracy::low>(-x, c);
  assert(s.hi() == t.hi() && s.lo() == t.lo());
  const P d = cuda::div<cuda::fpmp2_accuracy::mid>(x, c);
  const P e = cuda::div<cuda::fpmp2_accuracy::mid>(x, P(c));
  assert(d.hi() == e.hi() && d.lo() == e.lo());
}

template <typename P>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_accuracy<P>();
  test_scalar_minus_pair<P>();
  test_forms<P>();
  test_accuracy_selecting<P>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_type<cuda::fp32mp2_low>();
  test_type<cuda::fp32mp2_mid>();
  test_type<cuda::fp32mp2_high>();
  test_type<cuda::fp64mp2_low>();
  test_type<cuda::fp64mp2_mid>();
  test_type<cuda::fp64mp2_high>();
}

int main(int, char**)
{
  test();

  return 0;
}

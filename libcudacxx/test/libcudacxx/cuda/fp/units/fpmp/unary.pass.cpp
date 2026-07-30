// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 increment, decrement, negation and compound assignment.
//
//  fpmp2 is a class template, so an operator no test ever calls is never
//  instantiated and a broken body goes unnoticed. Every operator below is
//  therefore instantiated for both precisions and all four accuracy modes.
//
//  Most checks are identities against the binary operators (++x is x = x + 1,
//  x += y is x = x + y) so they hold in every accuracy mode; the exact
//  comparisons use values that a double-float and a double-double both
//  represent exactly.
//
//  fpmp2 also has scalar overloads of += and -= taking the underlying _FpType,
//  which take a different code path (__fpmp2_acc) than the fpmp2 overloads and
//  are checked separately.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Equality on the full multi-precision value, not just its double image, so a
// wrong low word cannot pass unnoticed.
template <class T>
TEST_FUNC bool same(const T& __a, const T& __b)
{
  return !(__a != __b);
}

// ---- increment / decrement ------------------------------------------------
template <class T>
TEST_FUNC bool test_incdec()
{
  const T start(2.5);
  const T one(1.0);

  bool ok = true;

  {
    T x   = start;
    T ret = ++x;
    ok    = ok && same(x, start + one); // side effect
    ok    = ok && same(ret, x); // prefix returns the new value
    ok    = ok && static_cast<double>(x) == 3.5;
  }
  {
    T x   = start;
    T ret = --x;
    ok    = ok && same(x, start - one);
    ok    = ok && same(ret, x);
    ok    = ok && static_cast<double>(x) == 1.5;
  }
  {
    T x   = start;
    T ret = x++;
    ok    = ok && same(x, start + one);
    ok    = ok && same(ret, start); // postfix returns the old value
    ok    = ok && static_cast<double>(x) == 3.5;
  }
  {
    T x   = start;
    T ret = x--;
    ok    = ok && same(x, start - one);
    ok    = ok && same(ret, start);
    ok    = ok && static_cast<double>(x) == 1.5;
  }

  // Prefix returns a reference to the object, not a copy.
  {
    T x = start;
    ++(++x);
    ok = ok && static_cast<double>(x) == 4.5;
  }

  // Crossing zero, where the (hi, lo) pair has to renormalize.
  {
    T x(0.5);
    --x;
    ok = ok && static_cast<double>(x) == -0.5;
    --x;
    ok = ok && static_cast<double>(x) == -1.5;
    ++x;
    ++x;
    ok = ok && static_cast<double>(x) == 0.5;
  }

  // The increment must land in the low word when the high word cannot hold it:
  // 2^30 + 1 is not representable in fp32, so a double-float that dropped the
  // low word would come back as 2^30 exactly.
  if constexpr (::cuda::std::is_same_v<T, fp32mp2> || ::cuda::std::is_same_v<T, fp64mp2>)
  {
    T x(1073741824.0); // 2^30
    ++x;
    ok = ok && static_cast<double>(x) == 1073741825.0;
    --x;
    ok = ok && static_cast<double>(x) == 1073741824.0;
  }

  return ok;
}

// ---- unary minus ---------------------------------------------------------
template <class T>
TEST_FUNC bool test_neg()
{
  bool ok = true;

  const T x(2.5);
  ok = ok && static_cast<double>(-x) == -2.5;
  ok = ok && static_cast<double>(-(-x)) == 2.5;
  ok = ok && static_cast<double>(x) == 2.5; // operand untouched
  ok = ok && same(-x, T(0.0) - x);

  const T y(-4.0);
  ok = ok && static_cast<double>(-y) == 4.0;

  // Both words must flip, not just the high one: negating a value with a
  // non-zero low word and adding the original has to cancel exactly.
  {
    const T z = T(1.0) / T(3.0);
    ok        = ok && static_cast<double>(z + (-z)) == 0.0;
    ok        = ok && same(-z, T(0.0) - z);
  }

  // Zero keeps its sign.
  ok = ok && ::cuda::std::signbit(static_cast<double>(-T(0.0)));

  return ok;
}

// ---- compound assignment -------------------------------------------------
template <class T>
TEST_FUNC bool test_compound()
{
  const T a(6.25);
  const T b(1.5);

  bool ok = true;

  {
    T x = a;
    x += b;
    ok = ok && same(x, a + b);
  }
  {
    T x = a;
    x -= b;
    ok = ok && same(x, a - b);
  }
  {
    T x = a;
    x *= b;
    ok = ok && same(x, a * b);
  }
  {
    T x = a;
    x /= b;
    ok = ok && same(x, a / b);
  }

  // Exact on powers of two in every mode.
  {
    T x(8.0);
    x += T(4.0);
    ok = ok && static_cast<double>(x) == 12.0;
    x -= T(2.0);
    ok = ok && static_cast<double>(x) == 10.0;
    x *= T(2.0);
    ok = ok && static_cast<double>(x) == 20.0;
    x /= T(4.0);
    ok = ok && static_cast<double>(x) == 5.0;
  }

  // The returned reference must be the object itself.
  {
    T x(1.0);
    (x += T(2.0)) += T(4.0);
    ok = ok && static_cast<double>(x) == 7.0;
  }

  return ok;
}

// ---- scalar += / -= ------------------------------------------------------
//
// These take _FpType and go through the accumulate path rather than the full
// add, so they are checked against the same operation spelled with fpmp2.
template <class T, class Scalar>
TEST_FUNC bool test_compound_scalar()
{
  bool ok = true;

  {
    T x(6.25);
    x += Scalar(1.5);
    ok = ok && static_cast<double>(x) == 7.75;
  }
  {
    T x(6.25);
    x -= Scalar(1.5);
    ok = ok && static_cast<double>(x) == 4.75;
  }

  // Accumulating a small value into a large one is the reason the scalar
  // overload exists: the low word has to keep what fp32 alone would lose.
  {
    T x(1.0);
    for (int i = 0; i < 4; ++i)
    {
      x += Scalar(1.0 / 1073741824.0); // 2^-30
    }
    ok = ok && static_cast<double>(x) == 1.0 + 4.0 / 1073741824.0;
  }

  return ok;
}

template <class T>
TEST_FUNC bool test_type()
{
  return test_incdec<T>() && test_neg<T>() && test_compound<T>();
}

TEST_FUNC void test()
{
  assert(test_type<fp32mp2>());
  assert(test_type<fp32mp2_low>());
  assert(test_type<fp32mp2_mid>());
  assert(test_type<fp32mp2_high>());

  assert(test_type<fp64mp2>());
  assert(test_type<fp64mp2_low>());
  assert(test_type<fp64mp2_mid>());
  assert(test_type<fp64mp2_high>());

  assert((test_compound_scalar<fp32mp2, float>()));
  assert((test_compound_scalar<fp64mp2, double>()));
}

int main(int, char**)
{
  test();

  return 0;
}

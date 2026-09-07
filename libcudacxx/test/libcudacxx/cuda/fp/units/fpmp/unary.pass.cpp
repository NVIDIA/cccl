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

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Equality on the full multi-precision value, not just its double image, so a
// wrong low word cannot pass unnoticed.
template <class T>
TEST_HOST_DEVICE_FUNC bool same(const T& __a, const T& __b)
{
  return !(__a != __b);
}

// ---- increment / decrement ------------------------------------------------
template <class T>
TEST_HOST_DEVICE_FUNC void test_incdec()
{
  const T start(2.5);
  const T one(1.0);

  {
    T x   = start;
    T ret = ++x;
    assert(same(x, start + one)); // side effect
    assert(same(ret, x)); // prefix returns the new value
    assert(static_cast<double>(x) == 3.5);
  }
  {
    T x   = start;
    T ret = --x;
    assert(same(x, start - one));
    assert(same(ret, x));
    assert(static_cast<double>(x) == 1.5);
  }
  {
    T x   = start;
    T ret = x++;
    assert(same(x, start + one));
    assert(same(ret, start)); // postfix returns the old value
    assert(static_cast<double>(x) == 3.5);
  }
  {
    T x   = start;
    T ret = x--;
    assert(same(x, start - one));
    assert(same(ret, start));
    assert(static_cast<double>(x) == 1.5);
  }

  // Prefix returns a reference to the object, not a copy.
  {
    T x = start;
    ++(++x);
    assert(static_cast<double>(x) == 4.5);
  }

  // Crossing zero, where the (hi, lo) pair has to renormalize.
  {
    T x(0.5);
    --x;
    assert(static_cast<double>(x) == -0.5);
    --x;
    assert(static_cast<double>(x) == -1.5);
    ++x;
    ++x;
    assert(static_cast<double>(x) == 0.5);
  }

  // The increment must land in the low word when the high word cannot hold it:
  // 2^30 + 1 is not representable in fp32, so a double-float that dropped the
  // low word would come back as 2^30 exactly.
  if constexpr (::cuda::std::is_same_v<T, cudax::fp32mp2> || ::cuda::std::is_same_v<T, cudax::fp64mp2>)
  {
    T x(1073741824.0); // 2^30
    ++x;
    assert(static_cast<double>(x) == 1073741825.0);
    --x;
    assert(static_cast<double>(x) == 1073741824.0);
  }
}

// ---- unary minus ---------------------------------------------------------
template <class T>
TEST_HOST_DEVICE_FUNC void test_neg()
{
  const T x(2.5);
  assert(static_cast<double>(-x) == -2.5);
  assert(static_cast<double>(-(-x)) == 2.5);
  assert(static_cast<double>(x) == 2.5); // operand untouched
  assert(same(-x, T(0.0) - x));

  const T y(-4.0);
  assert(static_cast<double>(-y) == 4.0);

  // Both words must flip, not just the high one: negating a value with a
  // non-zero low word and adding the original has to cancel exactly.
  {
    const T z = T(1.0) / T(3.0);
    assert(static_cast<double>(z + (-z)) == 0.0);
    assert(same(-z, T(0.0) - z));
  }

  // Zero keeps its sign.
  assert(::cuda::std::signbit(static_cast<double>(-T(0.0))));
}

// ---- compound assignment -------------------------------------------------
template <class T>
TEST_HOST_DEVICE_FUNC void test_compound()
{
  const T a(6.25);
  const T b(1.5);

  {
    T x = a;
    x += b;
    assert(same(x, a + b));
  }
  {
    T x = a;
    x -= b;
    assert(same(x, a - b));
  }
  {
    T x = a;
    x *= b;
    assert(same(x, a * b));
  }
  {
    T x = a;
    x /= b;
    assert(same(x, a / b));
  }

  // Exact on powers of two in every mode.
  {
    T x(8.0);
    x += T(4.0);
    assert(static_cast<double>(x) == 12.0);
    x -= T(2.0);
    assert(static_cast<double>(x) == 10.0);
    x *= T(2.0);
    assert(static_cast<double>(x) == 20.0);
    x /= T(4.0);
    assert(static_cast<double>(x) == 5.0);
  }

  // The returned reference must be the object itself.
  {
    T x(1.0);
    (x += T(2.0)) += T(4.0);
    assert(static_cast<double>(x) == 7.0);
  }
}

// ---- scalar += / -= ------------------------------------------------------
//
// These take _FpType and go through the accumulate path rather than the full
// add, so they are checked against the same operation spelled with fpmp2.
template <class T, class Scalar>
TEST_HOST_DEVICE_FUNC void test_compound_scalar()
{
  {
    T x(6.25);
    x += Scalar(1.5);
    assert(static_cast<double>(x) == 7.75);
  }
  {
    T x(6.25);
    x -= Scalar(1.5);
    assert(static_cast<double>(x) == 4.75);
  }

  // Accumulating a small value into a large one is the reason the scalar
  // overload exists: the low word has to keep what fp32 alone would lose.
  {
    T x(1.0);
    for (int i = 0; i < 4; ++i)
    {
      x += Scalar(1.0 / 1073741824.0); // 2^-30
    }
    assert(static_cast<double>(x) == 1.0 + 4.0 / 1073741824.0);
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_incdec<T>();
  test_neg<T>();
  test_compound<T>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_type<cudax::fp32mp2>();
  test_type<cudax::fp32mp2_low>();
  test_type<cudax::fp32mp2_mid>();
  test_type<cudax::fp32mp2_high>();

  test_type<cudax::fp64mp2>();
  test_type<cudax::fp64mp2_low>();
  test_type<cudax::fp64mp2_mid>();
  test_type<cudax::fp64mp2_high>();

  test_compound_scalar<cudax::fp32mp2, float>();
  test_compound_scalar<cudax::fp64mp2, double>();
}

int main(int, char**)
{
  test();

  return 0;
}

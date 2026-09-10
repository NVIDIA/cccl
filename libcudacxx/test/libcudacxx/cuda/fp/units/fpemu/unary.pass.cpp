// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpemu / fpemu_unpacked increment, decrement, negation and
//  compound assignment.
//
//  These operators had no test at all, which is how a body that assigns to
//  `this` instead of `*this` -- syntactically invalid, so it cannot compile --
//  survived a full CI run: fpemu is a class template, so a member function that
//  is never called is never instantiated and never diagnosed. Every operator
//  below is therefore instantiated for each accuracy mode, packed and unpacked.
//
//  Most checks are identities against the binary operators (++x is x = x + 1,
//  x += y is x = x + y) rather than hard-coded numbers, so they hold in every
//  accuracy mode; the exact comparisons use values every mode represents
//  exactly.
//
//  Operands come from opaque(), which launders the value through a volatile
//  object, because the runtime path is what is being validated here. When both
//  operands of an fpemu add are compile-time constants whose exponents are 64 or
//  more bits apart (which includes any add involving zero), the device
//  arithmetic is evaluated at compile time and the folded result differs from
//  what the same code computes on the GPU. Spelling the operands as literals
//  would test that compile-time path instead of the emulation.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/cassert>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Hide a value from the optimizer; see the note above.
TEST_FUNC double opaque(double __v)
{
  volatile double __t = __v;
  return __t;
}

// Equality of two emulated values through their double image.
template <class T>
TEST_FUNC bool same(const T& __a, const T& __b)
{
  return static_cast<double>(__a) == static_cast<double>(__b);
}

// ---- increment / decrement ------------------------------------------------
//
// Both the side effect and the returned value are checked. Postfix has to
// return the old value, which is what a naive `return *this` would get wrong.
//
// __correctly_rounded selects the checks that only the correctly rounded modes
// (def == high) have to satisfy. low is documented as half-mantissa accuracy over
// the normal range and mid as 1-2 ULP, and they do differ where it is visible:
// packed low returns the smallest normal instead of zero for 1.0 - 1.0, and mid
// perturbs 1.0 when a negligible operand is subtracted from it.
template <class T>
TEST_FUNC void test_incdec(bool __correctly_rounded)
{
  const T start(opaque(2.5));
  const T one(opaque(1.0));

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

  // Prefix returns a reference to the object itself, not a copy: incrementing
  // the result has to be visible in x.
  {
    T x = start;
    ++(++x);
    assert(static_cast<double>(x) == 4.5);
  }

  // A sequence has to land back where it started.
  {
    T x = start;
    ++x;
    ++x;
    --x;
    --x;
    assert(same(x, start));
  }

  // Counting up from zero: zero is the extreme case for the exponent alignment
  // inside the add, since it has no exponent to align with.
  {
    T x(opaque(0.0));
    ++x;
    assert(static_cast<double>(x) == 1.0);
    ++x;
    assert(static_cast<double>(x) == 2.0);
    --x;
    assert(static_cast<double>(x) == 1.0);
    if (__correctly_rounded)
    {
      --x;
      assert(static_cast<double>(x) == 0.0);
      --x;
      assert(static_cast<double>(x) == -1.0);
    }
  }
}

// ---- unary minus ---------------------------------------------------------
template <class T>
TEST_FUNC void test_neg(bool __correctly_rounded)
{
  const T x(opaque(2.5));
  assert(static_cast<double>(-x) == -2.5);
  assert(static_cast<double>(-(-x)) == 2.5);
  assert(static_cast<double>(x) == 2.5); // operand untouched

  const T y(opaque(-4.0));
  assert(static_cast<double>(-y) == 4.0);

  // Negation must agree with subtracting from zero.
  assert(same(-x, T(opaque(0.0)) - x));

  // Adding a value to its own negation cancels.
  if (__correctly_rounded)
  {
    assert(static_cast<double>(x + (-x)) == 0.0);
  }
}

// ---- compound assignment -------------------------------------------------
//
// Compared against the corresponding binary operator, so whatever the mode
// rounds to, the test still means "+= is +".
template <class T>
TEST_FUNC void test_compound(bool __correctly_rounded)
{
  const T a(opaque(6.25));
  const T b(opaque(1.5));

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
    T x(opaque(8.0));
    x += T(opaque(4.0));
    assert(static_cast<double>(x) == 12.0);
    x -= T(opaque(2.0));
    assert(static_cast<double>(x) == 10.0);
    x *= T(opaque(2.0));
    assert(static_cast<double>(x) == 20.0);
    x /= T(opaque(4.0));
    assert(static_cast<double>(x) == 5.0);
  }

  // The returned reference has to be the object, so a chained compound
  // assignment accumulates instead of updating a temporary.
  {
    T x(opaque(1.0));
    (x += T(opaque(2.0))) += T(opaque(4.0));
    assert(static_cast<double>(x) == 7.0);
  }

  // Adding a value far below the ulp of the accumulator has to leave it
  // unchanged rather than perturb it: the exponents here are more than 64 bits
  // apart, the widest alignment the add has to handle.
  if (__correctly_rounded)
  {
    T x(opaque(1.0));
    x += T(opaque(1e-300));
    assert(static_cast<double>(x) == 1.0);
    x -= T(opaque(1e-300));
    assert(static_cast<double>(x) == 1.0);
  }
}

template <class T>
TEST_FUNC void test_type(bool __correctly_rounded = true)
{
  test_incdec<T>(__correctly_rounded);
  test_neg<T>(__correctly_rounded);
  test_compound<T>(__correctly_rounded);
}

TEST_FUNC void test()
{
  // Packed. def == high are the correctly rounded modes; low and mid run the
  // same operators with the accuracy-dependent checks relaxed.
  test_type<cudax::fp64emu>();
  test_type<cudax::fp64emu_high>();
  test_type<cudax::fp64emu_low>(false);
  test_type<cudax::fp64emu_mid>(false);

  // Unpacked.
  test_type<cudax::fp64emu_unpacked>();
  test_type<cudax::fp64emu_unpacked_high>();
  test_type<cudax::fp64emu_unpacked_low>(false);
  test_type<cudax::fp64emu_unpacked_mid>(false);
}

int main(int, char**)
{
  test();

  return 0;
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp_custom increment, decrement, negation and compound assignment.
//
//  Only += was reachable from the existing fptool test, so the rest of this
//  surface was never instantiated. At the native field sizes fp64_custom is
//  a drop-in for double, so every result below is compared against the same
//  expression in double and must match exactly.
//
//  Negation is a sign-bit flip on the stored bits rather than a subtraction,
//  which is why it is also checked on zero and on a value whose mantissa the
//  precision callback would truncate.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC bool test_incdec()
{
  bool ok = true;

  {
    cudax::fp64_custom<> x(2.5);
    cudax::fp64_custom<> ret = ++x;
    ok                       = ok && static_cast<double>(x) == 3.5; // side effect
    ok                       = ok && static_cast<double>(ret) == 3.5; // prefix returns the new value
  }
  {
    cudax::fp64_custom<> x(2.5);
    cudax::fp64_custom<> ret = --x;
    ok                       = ok && static_cast<double>(x) == 1.5;
    ok                       = ok && static_cast<double>(ret) == 1.5;
  }
  {
    cudax::fp64_custom<> x(2.5);
    cudax::fp64_custom<> ret = x++;
    ok                       = ok && static_cast<double>(x) == 3.5;
    ok                       = ok && static_cast<double>(ret) == 2.5; // postfix returns the old value
  }
  {
    cudax::fp64_custom<> x(2.5);
    cudax::fp64_custom<> ret = x--;
    ok                       = ok && static_cast<double>(x) == 1.5;
    ok                       = ok && static_cast<double>(ret) == 2.5;
  }

  // Prefix returns a reference to the object, not a copy.
  {
    cudax::fp64_custom<> x(2.5);
    ++(++x);
    ok = ok && static_cast<double>(x) == 4.5;
  }

  // Crossing zero and coming back.
  {
    cudax::fp64_custom<> x(0.5);
    --x;
    ok = ok && static_cast<double>(x) == -0.5;
    --x;
    ok = ok && static_cast<double>(x) == -1.5;
    ++x;
    ++x;
    ok = ok && static_cast<double>(x) == 0.5;
  }

  return ok;
}

TEST_HOST_DEVICE_FUNC bool test_neg()
{
  bool ok = true;

  const cudax::fp64_custom<> x(2.5);
  ok = ok && static_cast<double>(-x) == -2.5;
  ok = ok && static_cast<double>(-(-x)) == 2.5;
  ok = ok && static_cast<double>(x) == 2.5; // operand untouched

  const cudax::fp64_custom<> y(-4.0);
  ok = ok && static_cast<double>(-y) == 4.0;

  // A sign-bit flip, so it also has to work on zero and on a non-dyadic value.
  ok = ok && ::cuda::std::signbit(static_cast<double>(-cudax::fp64_custom<>(0.0)));
  ok = ok && !::cuda::std::signbit(static_cast<double>(-cudax::fp64_custom<>(-0.0)));

  const cudax::fp64_custom<> z(0.1);
  ok = ok && static_cast<double>(-z) == -static_cast<double>(z);
  ok = ok && static_cast<double>(z + (-z)) == 0.0;

  return ok;
}

TEST_HOST_DEVICE_FUNC bool test_compound()
{
  bool ok = true;

  const double da = 6.25;
  const double db = 1.5;

  {
    cudax::fp64_custom<> x(da);
    x += cudax::fp64_custom<>(db);
    ok = ok && static_cast<double>(x) == da + db;
  }
  {
    cudax::fp64_custom<> x(da);
    x -= cudax::fp64_custom<>(db);
    ok = ok && static_cast<double>(x) == da - db;
  }
  {
    cudax::fp64_custom<> x(da);
    x *= cudax::fp64_custom<>(db);
    ok = ok && static_cast<double>(x) == da * db;
  }
  {
    cudax::fp64_custom<> x(da);
    x /= cudax::fp64_custom<>(db);
    ok = ok && static_cast<double>(x) == da / db;
  }

  // The returned reference must be the object itself.
  {
    cudax::fp64_custom<> x(1.0);
    (x += cudax::fp64_custom<>(2.0)) += cudax::fp64_custom<>(4.0);
    ok = ok && static_cast<double>(x) == 7.0;
  }

  // Accumulation must match double step for step.
  {
    cudax::fp64_custom<> x(0.0);
    double ref = 0.0;
    for (int i = 1; i <= 16; ++i)
    {
      x += cudax::fp64_custom<>(1.0 / static_cast<double>(i));
      ref += 1.0 / static_cast<double>(i);
    }
    ok = ok && static_cast<double>(x) == ref;
  }

  return ok;
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(test_incdec());
  assert(test_neg());
  assert(test_compound());
}

int main(int, char**)
{
  test();

  return 0;
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cuda::std math overloads for the fpmp2 multi-precision type.
//
//  The emulated math lives in cuda::experimental, but a qualified
//  cuda::std::<fn>(x) call suppresses ADL. Without dedicated overloads in
//  namespace cuda::std, such a call would silently narrow the fpmp2 argument to
//  double (via the implicit conversion) and compute a native-double result. This
//  test verifies that:
//    - cuda::std::sqrt / fma (from <cuda/fpmp>) and the standard <cmath>-named
//      functions (from <cuda/fpmp_math>) select the emulated implementation,
//    - the RETURN TYPE is the emulated type (not double) -- the compile-time
//      guard proving the double fallback overload was not chosen,
//    - mixed fpmp2 + built-in arithmetic operands are handled (fma),
//    - a few algebraic results match the expected values.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/fpmp_math>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC bool run_test()
{
  bool ok = true;

  using T = cudax::fp32mp2; // fpmp2<float, def>
  const T a(2.0f), b(3.0f), c(1.0f);

  // ---- Return-type guards (compile-time) ------------------------------------
  // If the emulated overloads were missing, cuda::std::<fn> would narrow the
  // fpmp2 argument to double and return double; these static_asserts fail to
  // compile in that case.
  // sqrt / fma live in <cuda/fpmp>:
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::sqrt(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(a, b, c)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(a, 3.0f, c)), T>); // mixed
  // Standard <cmath>-named functions live in <cuda/fpmp_math>:
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::exp(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::log(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::sin(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::cos(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::tanh(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::cbrt(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::pow(a, b)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::atan2(a, b)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fmax(a, b)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fmin(a, b)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fabs(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::copysign(a, b)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::ceil(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::floor(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::trunc(a)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::ldexp(a, 2)), T>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::ilogb(a)), int>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::llrint(a)), long long int>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::lround(a)), long int>);

  // fp64mp2 goes through the same overloads.
  using D = cudax::fp64mp2; // fpmp2<double, def>
  const D da(2.0), db(3.0), dc(1.0);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::sqrt(da)), D>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::fma(da, db, dc)), D>);
  static_assert(::cuda::std::is_same_v<decltype(::cuda::std::pow(da, db)), D>);

  // ---- Runtime sanity (algebraic; host + device exact) ----------------------
  const double s = static_cast<double>(::cuda::std::sqrt(T(4.0f)));
  ok             = ok && (s > 1.999 && s < 2.001);

  const double f = static_cast<double>(::cuda::std::fma(a, b, c)); // 2*3 + 1
  ok             = ok && (f > 6.999 && f < 7.001);

  const double fm = static_cast<double>(::cuda::std::fma(a, 3.0f, c)); // mixed, 2*3 + 1
  ok              = ok && (fm > 6.999 && fm < 7.001);

  const double mx = static_cast<double>(::cuda::std::fmax(a, b));
  ok              = ok && (mx > 2.999 && mx < 3.001);

  const double mn = static_cast<double>(::cuda::std::fmin(a, b));
  ok              = ok && (mn > 1.999 && mn < 2.001);

  const double av = static_cast<double>(::cuda::std::fabs(T(-3.0f)));
  ok              = ok && (av > 2.999 && av < 3.001);

  const double ds = static_cast<double>(::cuda::std::sqrt(da)); // sqrt(2)
  ok              = ok && (ds > 1.41420 && ds < 1.41422);

  const double df = static_cast<double>(::cuda::std::fma(da, db, dc)); // 2*3 + 1
  ok              = ok && (df > 6.999 && df < 7.001);

  const double dp = static_cast<double>(::cuda::std::pow(da, db)); // 2^3
  ok              = ok && (dp > 7.999 && dp < 8.001);

  return ok;
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(run_test());
}

int main(int, char**)
{
  test();

  return 0;
}

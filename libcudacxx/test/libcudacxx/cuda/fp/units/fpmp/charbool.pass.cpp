// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 char + bool operands mirror double.
//
//  bool and character types are excluded from __cccl_is_integer_v, so they are
//  routed through a dedicated converting constructor that upconverts to int32 and
//  reuses the existing int32 path. This mirrors double, for which `1.0 + true` and
//  `1.0 + 'a'` are valid.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// bool and character types must be constructible (mirrors double), while 128-bit
// integers remain deleted.
static_assert(::cuda::std::is_constructible_v<fp32mp2, bool>, "");
static_assert(::cuda::std::is_constructible_v<fp32mp2, char>, "");
static_assert(::cuda::std::is_constructible_v<fp32mp2, signed char>, "");
static_assert(::cuda::std::is_constructible_v<fp32mp2, unsigned char>, "");
static_assert(::cuda::std::is_constructible_v<fp32mp2, wchar_t>, "");
static_assert(::cuda::std::is_constructible_v<fp32mp2, char16_t>, "");
static_assert(::cuda::std::is_constructible_v<fp32mp2, char32_t>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, bool>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, char>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, signed char>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, unsigned char>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, wchar_t>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, char16_t>, "");
static_assert(::cuda::std::is_constructible_v<fp64mp2, char32_t>, "");

template <class FP>
_CCCL_HOST_DEVICE bool run_test()
{
  const double tol = 1e-5;
  bool ok          = true;

  // Pure construction from bool / char is exact (value <= 255 fits in int32/float).
  ok = ok && ((double) FP(true) == 1.0);
  ok = ok && ((double) FP(false) == 0.0);
  ok = ok && ((double) FP('a') == 97.0);
  ok = ok && ((double) FP((signed char) -5) == -5.0);
  ok = ok && ((double) FP((unsigned char) 200) == 200.0);

  // The wide character types are unsigned, and char32_t is as wide as int32_t: values
  // above 2^31 - 1 must stay positive rather than wrap through a signed widening.
  // Each of these is exactly representable, so the comparisons are exact.
  ok = ok && ((double) FP((char16_t) 0xFFFF) == 65535.0);
  ok = ok && ((double) FP((char32_t) 0x10FFFF) == 1114111.0);
  ok = ok && ((double) FP((char32_t) 0x80000000u) == 2147483648.0);
  ok = ok && ((double) FP((char32_t) 0xFFFFFFFFu) == 4294967295.0);

  // Mixed arithmetic mirrors double: 1.0 + true + 'a' == 1 + 1 + 97 == 99.
  {
    const double ref = 1.0 + true + 'a';
    FP a(1.0);
    FP r = FP(a + true) + 'a';
    ok   = ok && (::cuda::std::fabs((double) r - ref) <= tol);
  }

  // char on the left-hand side of a mixed op.
  {
    const double ref = 'a' + 2.0;
    FP b(2.0);
    ok = ok && (::cuda::std::fabs((double) FP('a' + b) - ref) <= tol);
  }

  // bool used as a multiplicative mask (true -> keep, false -> zero).
  {
    FP c(3.5);
    ok = ok && (::cuda::std::fabs((double) FP(c * true) - 3.5) <= tol);
    ok = ok && (::cuda::std::fabs((double) FP(c * false) - 0.0) <= tol);
  }

  return ok;
}

TEST_FUNC void test()
{
  assert(run_test<fp32mp2>());
  assert(run_test<fp64mp2>());
}

int main(int, char**)
{
  test();

  return 0;
}

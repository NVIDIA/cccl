// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: mixed-type accuracy-explicit overloads.
//
//  Companion to fpmp accuracy. Tests the mixed-type overloads of add<m>, sub<m>,
//  mul<m>, div<m>, fma<m>, mad<m> that accept any mix of fpmp2 and built-in
//  arithmetic operands (at least one fpmp2). Each mixed-type call must be
//  bit-identical to the strict all-fpmp2 call where the scalar(s) are wrapped in
//  the participating type; result-type preservation is pinned by static_asserts.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Result type must match the participating fpmp2 type, regardless of operand
// order or which argument is the scalar.
static_assert(
  ::cuda::std::is_same<decltype(add<fpmp2_accuracy::high>(::cuda::std::declval<fp32mp2>(), 1.0f)), fp32mp2>::value,
  "add(mp2_def, float) must return mp2_def");
static_assert(::cuda::std::is_same<decltype(sub<fpmp2_accuracy::low>(1.0f, ::cuda::std::declval<fp32mp2_low>())),
                                   fp32mp2_low>::value,
              "sub(float, mp2_low) must return mp2_low");
static_assert(::cuda::std::is_same<decltype(mul<fpmp2_accuracy::high>(::cuda::std::declval<fp32mp2_high>(), 2)),
                                   fp32mp2_high>::value,
              "mul(mp2_high, int) must return mp2_high");
static_assert(::cuda::std::is_same<decltype(fma<fpmp2_accuracy::low>(1.0f, ::cuda::std::declval<fp32mp2_low>(), 2.0f)),
                                   fp32mp2_low>::value,
              "fma(float, mp2_low, float) must return mp2_low");
static_assert(
  ::cuda::std::is_same<decltype(mad<fpmp2_accuracy::def>(1.0f, 2.0f, ::cuda::std::declval<fp32mp2>())), fp32mp2>::value,
  "mad(float, float, mp2_def) must return mp2_def");

// Each mixed-type call must equal the strict form (scalars wrapped in the fpmp2
// type) bit-for-bit.
_CCCL_HOST_DEVICE void run_test()
{
  using ff = fp32mp2_low;
  ff a(1.234567890f), b(2.345678901f);
  const float s = 0.5f;
  const float t = 3.0f;

  // Binary, both argument orders.
  assert((double) add<fpmp2_accuracy::high>(a, s) == (double) add<fpmp2_accuracy::high>(a, ff(s)));
  assert((double) add<fpmp2_accuracy::high>(s, a) == (double) add<fpmp2_accuracy::high>(ff(s), a));
  assert((double) sub<fpmp2_accuracy::high>(a, s) == (double) sub<fpmp2_accuracy::high>(a, ff(s)));
  assert((double) sub<fpmp2_accuracy::high>(s, a) == (double) sub<fpmp2_accuracy::high>(ff(s), a));
  assert((double) mul<fpmp2_accuracy::low>(a, s) == (double) mul<fpmp2_accuracy::low>(a, ff(s)));
  assert((double) mul<fpmp2_accuracy::low>(s, a) == (double) mul<fpmp2_accuracy::low>(ff(s), a));
  assert((double) div<fpmp2_accuracy::def>(a, s) == (double) div<fpmp2_accuracy::def>(a, ff(s)));
  assert((double) div<fpmp2_accuracy::def>(s, a) == (double) div<fpmp2_accuracy::def>(ff(s), a));

  // Ternary fma: scalar in every position.
  assert((double) fma<fpmp2_accuracy::high>(a, s, t) == (double) fma<fpmp2_accuracy::high>(a, ff(s), ff(t)));
  assert((double) fma<fpmp2_accuracy::high>(s, a, t) == (double) fma<fpmp2_accuracy::high>(ff(s), a, ff(t)));
  assert((double) fma<fpmp2_accuracy::high>(s, t, a) == (double) fma<fpmp2_accuracy::high>(ff(s), ff(t), a));

  // Ternary mad: one scalar, two fpmp2 operands.
  assert((double) mad<fpmp2_accuracy::low>(a, b, s) == (double) mad<fpmp2_accuracy::low>(a, b, ff(s)));
  assert((double) mad<fpmp2_accuracy::low>(a, s, b) == (double) mad<fpmp2_accuracy::low>(a, ff(s), b));
  assert((double) mad<fpmp2_accuracy::low>(s, a, b) == (double) mad<fpmp2_accuracy::low>(ff(s), a, b));
}

TEST_FUNC void test()
{
  run_test();
}

int main(int, char**)
{
  test();

  return 0;
}

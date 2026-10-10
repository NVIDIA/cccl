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

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Result type must match the participating fpmp2 type, regardless of operand
// order or which argument is the scalar.
static_assert(
  ::cuda::std::is_same<decltype(cudax::add<cudax::fpmp2_accuracy::high>(::cuda::std::declval<cudax::fp32mp2>(), 1.0f)),
                       cudax::fp32mp2>::value,
  "add(mp2_def, float) must return mp2_def");
static_assert(::cuda::std::is_same<
                decltype(cudax::sub<cudax::fpmp2_accuracy::low>(1.0f, ::cuda::std::declval<cudax::fp32mp2_low>())),
                cudax::fp32mp2_low>::value,
              "sub(float, mp2_low) must return mp2_low");
static_assert(
  ::cuda::std::is_same<decltype(cudax::mul<cudax::fpmp2_accuracy::high>(::cuda::std::declval<cudax::fp32mp2_high>(), 2)),
                       cudax::fp32mp2_high>::value,
  "mul(mp2_high, int) must return mp2_high");
static_assert(
  ::cuda::std::is_same<
    decltype(cudax::fma<cudax::fpmp2_accuracy::low>(1.0f, ::cuda::std::declval<cudax::fp32mp2_low>(), 2.0f)),
    cudax::fp32mp2_low>::value,
  "fma(float, mp2_low, float) must return mp2_low");
static_assert(::cuda::std::is_same<
                decltype(cudax::mad<cudax::fpmp2_accuracy::def>(1.0f, 2.0f, ::cuda::std::declval<cudax::fp32mp2>())),
                cudax::fp32mp2>::value,
              "mad(float, float, mp2_def) must return mp2_def");

// Each mixed-type call must equal the strict form (scalars wrapped in the fpmp2
// type) bit-for-bit.
TEST_HOST_DEVICE_FUNC void run_test()
{
  using ff = cudax::fp32mp2_low;
  ff a(1.234567890f), b(2.345678901f);
  const float s = 0.5f;
  const float t = 3.0f;

  // Binary, both argument orders.
  assert((double) cudax::add<cudax::fpmp2_accuracy::high>(a, s)
         == (double) cudax::add<cudax::fpmp2_accuracy::high>(a, ff(s)));
  assert((double) cudax::add<cudax::fpmp2_accuracy::high>(s, a)
         == (double) cudax::add<cudax::fpmp2_accuracy::high>(ff(s), a));
  assert((double) cudax::sub<cudax::fpmp2_accuracy::high>(a, s)
         == (double) cudax::sub<cudax::fpmp2_accuracy::high>(a, ff(s)));
  assert((double) cudax::sub<cudax::fpmp2_accuracy::high>(s, a)
         == (double) cudax::sub<cudax::fpmp2_accuracy::high>(ff(s), a));
  assert((double) cudax::mul<cudax::fpmp2_accuracy::low>(a, s)
         == (double) cudax::mul<cudax::fpmp2_accuracy::low>(a, ff(s)));
  assert((double) cudax::mul<cudax::fpmp2_accuracy::low>(s, a)
         == (double) cudax::mul<cudax::fpmp2_accuracy::low>(ff(s), a));
  assert((double) cudax::div<cudax::fpmp2_accuracy::def>(a, s)
         == (double) cudax::div<cudax::fpmp2_accuracy::def>(a, ff(s)));
  assert((double) cudax::div<cudax::fpmp2_accuracy::def>(s, a)
         == (double) cudax::div<cudax::fpmp2_accuracy::def>(ff(s), a));

  // Ternary fma: scalar in every position.
  assert((double) cudax::fma<cudax::fpmp2_accuracy::high>(a, s, t)
         == (double) cudax::fma<cudax::fpmp2_accuracy::high>(a, ff(s), ff(t)));
  assert((double) cudax::fma<cudax::fpmp2_accuracy::high>(s, a, t)
         == (double) cudax::fma<cudax::fpmp2_accuracy::high>(ff(s), a, ff(t)));
  assert((double) cudax::fma<cudax::fpmp2_accuracy::high>(s, t, a)
         == (double) cudax::fma<cudax::fpmp2_accuracy::high>(ff(s), ff(t), a));

  // Ternary mad: one scalar, two fpmp2 operands.
  assert((double) cudax::mad<cudax::fpmp2_accuracy::low>(a, b, s)
         == (double) cudax::mad<cudax::fpmp2_accuracy::low>(a, b, ff(s)));
  assert((double) cudax::mad<cudax::fpmp2_accuracy::low>(a, s, b)
         == (double) cudax::mad<cudax::fpmp2_accuracy::low>(a, ff(s), b));
  assert((double) cudax::mad<cudax::fpmp2_accuracy::low>(s, a, b)
         == (double) cudax::mad<cudax::fpmp2_accuracy::low>(ff(s), a, b));
}

TEST_HOST_DEVICE_FUNC void test()
{
  run_test();
}

int main(int, char**)
{
  test();

  return 0;
}

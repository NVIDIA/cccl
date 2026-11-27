// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2_stat parity with fpmp2 when CCCL_FPMP_EXPLICIT_CASTS=0.
//
//  With implicit casts a double reaches fp32mp2 through the two-limb conversion;
//  fpmp2_stat must convert it the same way rather than narrowing it to float.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_ENABLE_FPTOOL, CCCL_FPMP_EXPLICIT_CASTS=0
// note: fptool is opt-in, because its types carry mutable state at namespace scope;
// see <cuda/__fp/fptool_common.h>

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class A, class B>
TEST_HOST_DEVICE_FUNC bool same(const A& a, const B& b)
{
  return a.hi() == b.hi() && a.lo() == b.lo();
}

TEST_HOST_DEVICE_FUNC void test()
{
  using base_t = cuda::fp32mp2;
  using stat_t = cuda::fp32mp2_stat;

  const double wide = 1.0 + 0x1p-30; // needs both float limbs
  base_t b(1.25f);
  stat_t s(1.25f);
  b += wide;
  s += wide;
  assert(same(b, s));
  b *= wide;
  s *= wide;
  assert(same(b, s));
  b /= wide;
  s /= wide;
  assert(same(b, s));
  b -= wide;
  s -= wide;
  assert(same(b, s));
  assert(same(b * wide, s * wide));
  assert(same(wide / b, wide / s));
}

int main(int, char**)
{
  test();
  return 0;
}

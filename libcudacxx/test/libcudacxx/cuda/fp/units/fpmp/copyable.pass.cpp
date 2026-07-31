// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 trivial copyability + volatile round-trip.
//
//  A compile-time static_assert checks that fp32mp2 is trivially copyable; the
//  runtime run_test() confirms a value survives a round-trip through a volatile
//  object.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

static_assert(::cuda::std::is_trivially_copyable<fp32mp2>::value, "fp32mp2 must be trivially copyable");

// Assign through a volatile object and confirm the value is preserved. The value is
// read back through the volatile copy constructor and compared as an fpmp2: comparing
// the volatile object directly would have gone through operator double(), which comes
// down to the rounded double image rather than the pair that was stored.
_CCCL_HOST_DEVICE void run_test()
{
  volatile fp32mp2 vx[1];
  fp32mp2 x[1] = {fp32mp2(1.0e+20)};
  vx[0]        = x[0];

  const fp32mp2 read_back = vx[0];
  assert(!(read_back != x[0]));
  assert(read_back.hi() == x[0].hi());
  assert(read_back.lo() == x[0].lo());
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

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

static_assert(std::is_trivially_copyable<fp32mp2>::value, "fp32mp2 must be trivially copyable");

// Assign through a volatile object and confirm the value is preserved.
_CCCL_HOST_DEVICE bool run_test()
{
  volatile fp32mp2 vx[1];
  fp32mp2 x[1] = {fp32mp2(1.0e+20)};
  vx[0]        = x[0];
  return !(vx[0] != x[0]);
}

TEST_FUNC void test()
{
  assert(run_test());
}

int main(int, char**)
{
  test();

  return 0;
}

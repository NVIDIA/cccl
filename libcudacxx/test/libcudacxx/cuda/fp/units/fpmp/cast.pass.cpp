// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: idempotency of sequential double <-> fp32mp2 casts.
//
//  Verifies that the conversion double -> fp32mp2 -> double stabilizes after the
//  first round trip: repeating it (d1 -> d2 -> d3) leaves the value unchanged
//  (d1 == d2 == d3) across a range of magnitudes.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using ffloat = cudax::fp32mp2;

// double -> fp32mp2 -> double must be idempotent after the first round trip.
TEST_HOST_DEVICE_FUNC void run_test()
{
  const double tv[10] = {
    1234.567890123456777,
    0.1234567890123456777,
    1.23456789012345e-10,
    1.2345678901234567891,
    -9.8765432109876543211,
    3.1415926535897932383,
    -1.98765432109876e-15,
    1.1111111111111111e10,
    -9.8765432109876543211e14,
    2.718281828459045235,
  };

  for (int i = 0; i < 10; i++)
  {
    const double d1 = (double) ffloat(tv[i]);
    const double d2 = (double) ffloat(d1);
    const double d3 = (double) ffloat(d2);
    assert((d1 == d2) && (d2 == d3));
  }
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

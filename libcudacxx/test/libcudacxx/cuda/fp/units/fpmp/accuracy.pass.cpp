// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: accuracy-explicit arithmetic free functions.
//
//  Tests the accuracy-explicit free functions add<m>, sub<m>, mul<m>, div<m>,
//  fma<m>, mad<m> which override the arithmetic accuracy for a single operation
//  without changing the result type. Each must be bit-identical (tol == 0) to the
//  operator-based computation on the equivalently-tagged type, including
//  cross-accuracy calls; a large-cancellation case additionally checks that the
//  high-accuracy sum is at least as accurate as the low-accuracy one.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC void run_test()
{
  // Near-cancelling pair for add/sub; normal-range values for mul/div/fma/mad.
  const double a = -2.7059461654979244e+033;
  const double b = +2.7059454398538426e+033;
  const double x = 1.234567890123456;
  const double y = 2.345678901234567;
  const double z = 0.567890123456789;

  cudax::fp32mp2 ad(a), bd(b), xd(x), yd(y), zd(z);
  cudax::fp32mp2_low af(a), bf(b), xf(x), yf(y), zf(z);
  cudax::fp32mp2_high aa(a), ba(b), xa(x), ya(y), za(z);

  // add<m> vs operator+ on the equivalently-tagged type.
  assert((double) cudax::add<cudax::fpmp2_accuracy::def>(ad, bd) == (double) (ad + bd));
  assert((double) cudax::add<cudax::fpmp2_accuracy::low>(ad, bd) == (double) (af + bf));
  assert((double) cudax::add<cudax::fpmp2_accuracy::high>(ad, bd) == (double) (aa + ba));

  // sub<m> vs operator-.
  assert((double) cudax::sub<cudax::fpmp2_accuracy::def>(ad, -bd) == (double) (ad - (-bd)));
  assert((double) cudax::sub<cudax::fpmp2_accuracy::low>(ad, -bd) == (double) (af - (-bf)));
  assert((double) cudax::sub<cudax::fpmp2_accuracy::high>(ad, -bd) == (double) (aa - (-ba)));

  // mul<m> vs operator*.
  assert((double) cudax::mul<cudax::fpmp2_accuracy::def>(xd, yd) == (double) (xd * yd));
  assert((double) cudax::mul<cudax::fpmp2_accuracy::low>(xd, yd) == (double) (xf * yf));
  // There is no dedicated accurate multiplication, so high resolves to the default path.
  assert((double) cudax::mul<cudax::fpmp2_accuracy::high>(xd, yd) == (double) (xd * yd));

  // div<m> vs operator/.
  assert((double) cudax::div<cudax::fpmp2_accuracy::def>(xd, yd) == (double) (xd / yd));
  assert((double) cudax::div<cudax::fpmp2_accuracy::low>(xd, yd) == (double) (xf / yf));
  assert((double) cudax::div<cudax::fpmp2_accuracy::high>(xd, yd) == (double) (xa / ya));

  // fma<m> vs fma().
  assert((double) cudax::fma<cudax::fpmp2_accuracy::def>(xd, yd, zd) == (double) fma(xd, yd, zd));
  assert((double) cudax::fma<cudax::fpmp2_accuracy::low>(xd, yd, zd) == (double) fma(xf, yf, zf));
  assert((double) cudax::fma<cudax::fpmp2_accuracy::high>(xd, yd, zd) == (double) fma(xa, ya, za));

  // mad<m> vs mad().
  assert((double) cudax::mad<cudax::fpmp2_accuracy::def>(xd, yd, zd) == (double) cudax::mad(xd, yd, zd));
  assert((double) cudax::mad<cudax::fpmp2_accuracy::low>(xd, yd, zd) == (double) cudax::mad(xf, yf, zf));
  assert((double) cudax::mad<cudax::fpmp2_accuracy::high>(xd, yd, zd) == (double) cudax::mad(xa, ya, za));

  // Cross-accuracy: op<m> on a differently-tagged operand.
  assert((double) cudax::sub<cudax::fpmp2_accuracy::high>(af, -bf) == (double) (aa - (-ba)));
  assert((double) cudax::add<cudax::fpmp2_accuracy::def>(af, bf) == (double) (ad + bd));
  assert((double) cudax::fma<cudax::fpmp2_accuracy::high>(xf, yf, zf) == (double) fma(xa, ya, za));

  // Large cancellation: high accuracy must be at least as good as low.
  {
    cudax::fp32mp2_low ca(a), cb(b);
    const double exact = a + b;
    const double efast = ::cuda::std::fabs((double) cudax::add<cudax::fpmp2_accuracy::low>(ca, cb) - exact);
    const double eacc  = ::cuda::std::fabs((double) cudax::add<cudax::fpmp2_accuracy::high>(ca, cb) - exact);
    assert(eacc <= efast);
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

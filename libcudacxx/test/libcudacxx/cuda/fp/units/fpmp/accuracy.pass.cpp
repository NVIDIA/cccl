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

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

_CCCL_HOST_DEVICE bool run_test()
{
  // Near-cancelling pair for add/sub; normal-range values for mul/div/fma/mad.
  const double a = -2.7059461654979244e+033;
  const double b = +2.7059454398538426e+033;
  const double x = 1.234567890123456;
  const double y = 2.345678901234567;
  const double z = 0.567890123456789;

  fp32mp2 ad(a), bd(b), xd(x), yd(y), zd(z);
  fp32mp2_low af(a), bf(b), xf(x), yf(y), zf(z);
  fp32mp2_high aa(a), ba(b), xa(x), ya(y), za(z);

  bool ok = true;

  // add<m> vs operator+ on the equivalently-tagged type.
  ok = ok && ((double) add<fpmp2_accuracy::def>(ad, bd) == (double) (ad + bd));
  ok = ok && ((double) add<fpmp2_accuracy::low>(ad, bd) == (double) (af + bf));
  ok = ok && ((double) add<fpmp2_accuracy::high>(ad, bd) == (double) (aa + ba));

  // sub<m> vs operator-.
  ok = ok && ((double) sub<fpmp2_accuracy::def>(ad, -bd) == (double) (ad - (-bd)));
  ok = ok && ((double) sub<fpmp2_accuracy::low>(ad, -bd) == (double) (af - (-bf)));
  ok = ok && ((double) sub<fpmp2_accuracy::high>(ad, -bd) == (double) (aa - (-ba)));

  // mul<m> vs operator*.
  ok = ok && ((double) mul<fpmp2_accuracy::def>(xd, yd) == (double) (xd * yd));
  ok = ok && ((double) mul<fpmp2_accuracy::low>(xd, yd) == (double) (xf * yf));
#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
  ok = ok && ((double) mul<fpmp2_accuracy::high>(xd, yd) == (double) (xa * ya));
#else
  ok = ok && ((double) mul<fpmp2_accuracy::high>(xd, yd) == (double) (xd * yd));
#endif

  // div<m> vs operator/.
  ok = ok && ((double) div<fpmp2_accuracy::def>(xd, yd) == (double) (xd / yd));
  ok = ok && ((double) div<fpmp2_accuracy::low>(xd, yd) == (double) (xf / yf));
#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
  ok = ok && ((double) div<fpmp2_accuracy::high>(xd, yd) == (double) (xa / ya));
#else
  ok = ok && ((double) div<fpmp2_accuracy::high>(xd, yd) == (double) (xd / yd));
#endif

  // fma<m> vs fma().
  ok = ok && ((double) fma<fpmp2_accuracy::def>(xd, yd, zd) == (double) fma(xd, yd, zd));
  ok = ok && ((double) fma<fpmp2_accuracy::low>(xd, yd, zd) == (double) fma(xf, yf, zf));
  ok = ok && ((double) fma<fpmp2_accuracy::high>(xd, yd, zd) == (double) fma(xa, ya, za));

  // mad<m> vs mad().
  ok = ok && ((double) mad<fpmp2_accuracy::def>(xd, yd, zd) == (double) mad(xd, yd, zd));
  ok = ok && ((double) mad<fpmp2_accuracy::low>(xd, yd, zd) == (double) mad(xf, yf, zf));
  ok = ok && ((double) mad<fpmp2_accuracy::high>(xd, yd, zd) == (double) mad(xa, ya, za));

  // Cross-accuracy: op<m> on a differently-tagged operand.
  ok = ok && ((double) sub<fpmp2_accuracy::high>(af, -bf) == (double) (aa - (-ba)));
  ok = ok && ((double) add<fpmp2_accuracy::def>(af, bf) == (double) (ad + bd));
  ok = ok && ((double) fma<fpmp2_accuracy::high>(xf, yf, zf) == (double) fma(xa, ya, za));

  // Large cancellation: high accuracy must be at least as good as low.
  {
    fp32mp2_low ca(a), cb(b);
    const double exact = a + b;
    const double efast = ::cuda::std::fabs((double) add<fpmp2_accuracy::low>(ca, cb) - exact);
    const double eacc  = ::cuda::std::fabs((double) add<fpmp2_accuracy::high>(ca, cb) - exact);
    ok                 = ok && (eacc <= efast);
  }

  return ok;
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

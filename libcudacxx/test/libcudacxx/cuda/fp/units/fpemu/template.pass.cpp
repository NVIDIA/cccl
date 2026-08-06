// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu accuracy template parameter selection.
//
//  Demonstrates selecting the emulation accuracy at compile time via
//  fpemu<double, m>. The expression ((x + x) * x - x) / (x + c) is evaluated with
//  high / def / low accuracy and each result is checked against the native double
//  reference with an accuracy-appropriate tolerance (tight for high, relaxed for
//  low).
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Evaluate ((x + x) * x - x) / (x + c) with a chosen accuracy, using the builtins
// (which deduce the accuracy from their fpemu<double, m> argument types).
template <cudax::fpemu_accuracy m>
TEST_HOST_DEVICE_FUNC void test(double x0, double ref, double tol)
{
  cudax::fpemu<double, m> x = x0;
  cudax::fpemu<double, m> c = 0.001;
  const auto result = static_cast<double>(__ddiv_rn(__dsub_rn(__dmul_rn(__dadd_rn(x, x), x), x), __dadd_rn(x, c)));

  assert(::cuda::std::fabs(result - ref) <= tol);
}

TEST_HOST_DEVICE_FUNC void test(double x0)
{
  const double c   = 0.001;
  const double ref = ((x0 + x0) * x0 - x0) / (x0 + c);

  test<cudax::fpemu_accuracy::high>(x0, ref, 1e-12);
  test<cudax::fpemu_accuracy::def>(x0, ref, 1e-10);
  test<cudax::fpemu_accuracy::low>(x0, ref, 1e-4);
}

int main(int, char**)
{
  constexpr double x0 = -0x1.57f1782782a8ap-1;
  test(x0);

  return 0;
}

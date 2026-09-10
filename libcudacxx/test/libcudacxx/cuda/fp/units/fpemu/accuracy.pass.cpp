// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu accuracy levels.
//
//  Exercises the full set of fpemu accuracy selectors (high / def / low) through
//  the builtin ops (__dadd_rn, __dmul_rn, __dsub_rn, __ddiv_rn, __fma_rn,
//  __dsqrt_rn). The builtins deduce the accuracy level from the argument type, so
//  no explicit template parameters are needed. The result is checked for basic
//  sanity (finite, non-zero, not absurdly large).
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Computes across all three accuracy levels and verifies the aggregate result is
// finite, non-zero and reasonably bounded. Returns true on success.
TEST_HOST_DEVICE_FUNC void test(double x)
{
  // high accuracy: builtins deduce the accuracy level from the argument type.
  cudax::fp64emu_high acc_x = x;
  auto acc_r                = cudax::__dadd_rn(acc_x, acc_x);
  acc_r                     = cudax::__dmul_rn(acc_r, acc_x);
  acc_r                     = cudax::__dsub_rn(acc_r, acc_x);
  acc_r                     = cudax::__ddiv_rn(acc_r, acc_x);
  acc_r                     = cudax::__fma_rn(acc_r, acc_x, acc_x);
  acc_r                     = cudax::__dsqrt_rn(acc_r);

  // default accuracy: fp64emu is fpemu<double, fpemu_accuracy::def> (== high).
  cudax::fp64emu def_x = x;
  auto def_r           = cudax::__dadd_rn(def_x, def_x);
  def_r                = cudax::__dmul_rn(def_r, def_x);
  def_r                = cudax::__dsub_rn(def_r, def_x);
  def_r                = cudax::__ddiv_rn(def_r, def_x);
  def_r                = cudax::__fma_rn(def_r, def_x, def_x);
  def_r                = cudax::__dsqrt_rn(def_r);

  // low accuracy: builtins deduce the accuracy level from the argument type.
  cudax::fp64emu_low fast_x = x;
  auto fast_r               = cudax::__dadd_rn(fast_x, fast_x);
  fast_r                    = cudax::__dmul_rn(fast_r, fast_x);
  fast_r                    = cudax::__dsub_rn(fast_r, fast_x);
  fast_r                    = cudax::__ddiv_rn(fast_r, fast_x);

  const double r = (double) acc_r + (double) def_r + (double) fast_r;

  // Sanity: not NaN, not zero, and not absurdly large.
  assert(r == r);
  assert(r != 0.0);
  assert(cuda::std::fabs(r) < 1e20);
}

int main(int, char**)
{
  const double x = 1.2345;
  test(x);

  return 0;
}

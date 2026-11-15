// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: the cast-based double -> fp32mp2 conversion, which
//  CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=0 selects in place of the integer one.
//
//  A double that is infinite, NaN or beyond FLT_MAX converts to a pair with a
//  zero low limb, and so do the results of the functions that compute in double
//  and convert back: the fpmp2_accuracy::high norm3d, norm4d, rnorm3d and
//  rnorm4d. Finite values in range convert to the normalized pair of the value.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=0

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE_FUNC bool is_inf_pair(T r, bool negative)
{
  return cuda::std::isinf(r.hi()) && cuda::std::signbit(r.hi()) == negative && r.lo() == 0.0f;
}

template <class T>
TEST_HOST_DEVICE_FUNC void test()
{
  const double inf   = cuda::std::numeric_limits<double>::infinity();
  const float fltmax = (cuda::std::numeric_limits<float>::max)();

  assert(is_inf_pair(T(inf), false) && is_inf_pair(T(-inf), true));
  assert(is_inf_pair(T(1e300), false) && is_inf_pair(T(-1e300), true));
  assert(is_inf_pair(T(2.0 * fltmax), false));
  const T nan(cuda::std::numeric_limits<double>::quiet_NaN());
  assert(cuda::std::isnan(nan.hi()) && nan.lo() == 0.0f);

  // in range: the pair holds the double, hi the nearest float
  const double wide = 1.0 + 0x1p-30;
  const T w(wide);
  assert(w.hi() == 1.0f && w.lo() == 0x1p-30f);
  const T top(static_cast<double>(fltmax));
  assert(top.hi() == fltmax && top.lo() == 0.0f);
  assert(static_cast<double>(T(-0x1.23456789abp-70)) == -0x1.23456789abp-70);

  // the norms that overflow, and the reciprocal norms of zero operands
  const T big(fltmax), zero(0.0f);
  assert(is_inf_pair(norm3d(big, big, big), false));
  assert(is_inf_pair(norm4d(big, zero, zero, big), false));
  assert(is_inf_pair(rnorm3d(zero, zero, zero), false));
  assert(is_inf_pair(rnorm4d(zero, zero, zero, zero), false));
}

int main(int, char**)
{
  test<cuda::fp32mp2>();
  test<cuda::fp32mp2_low>();
  test<cuda::fp32mp2_high>();
  return 0;
}

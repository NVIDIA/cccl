// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpemu / fpemu_unpacked char + bool operands mirror double.
//
//  bool and character types are excluded from __cccl_is_integer_v, so they are
//  routed through a dedicated converting constructor that upconverts to int32 and
//  reuses the existing int32 path. This mirrors double, for which `1.0 + true` and
//  `1.0 + 'a'` are valid.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// bool and character types must be constructible (mirrors double), while 128-bit
// integers remain deleted.
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, bool>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, char>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, signed char>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, unsigned char>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, wchar_t>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu_unpacked<double>, bool>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu_unpacked<double>, char>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu_unpacked<double>, signed char>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu_unpacked<double>, unsigned char>);

template <class FP>
TEST_HOST_DEVICE_FUNC void test()
{
  const double tol = 1e-10;

  // Pure construction from bool / char is exact (value <= 255 fits in int32/double).
  assert((double) FP(true) == 1.0);
  assert((double) FP(false) == 0.0);
  assert((double) FP('a') == 97.0);
  assert((double) FP((signed char) -5) == -5.0);
  assert((double) FP((unsigned char) 200) == 200.0);

  // Mixed arithmetic mirrors double: 1.0 + true + 'a' == 1 + 1 + 97 == 99.
  {
    const double ref = 1.0 + true + 'a';
    FP a(1.0);
    FP r = FP(a + true) + 'a';
    assert(cuda::std::fabs((double) r - ref) <= tol);
  }

  // char on the left-hand side of a mixed op.
  {
    const double ref = 'a' + 2.0;
    FP b(2.0);
    assert(cuda::std::fabs((double) FP('a' + b) - ref) <= tol);
  }

  // bool used as a multiplicative mask (true -> keep, false -> zero).
  {
    FP c(3.5);
    assert(cuda::std::fabs((double) FP(c * true) - 3.5) <= tol);
    assert(cuda::std::fabs((double) FP(c * false) - 0.0) <= tol);
  }
}

TEST_HOST_DEVICE_FUNC void test()
{
  test<cudax::fpemu<double>>();
  test<cudax::fpemu_unpacked<double>>();
}

int main(int, char**)
{
  test();

  return 0;
}

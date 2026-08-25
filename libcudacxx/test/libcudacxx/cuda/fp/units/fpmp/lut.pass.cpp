// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: compile-time double -> fp32mp2 lookup table.
//
//  A constexpr fp32mp2 lookup table is initialized directly from double literals
//  (compile-time conversion, zero runtime overhead). The test loads pairs of
//  entries, multiplies them, and verifies the fp32mp2 products track the double
//  reference within tolerance, plus that each stored entry round-trips to its
//  literal within tolerance.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using ffloat = cudax::fp32mp2;

// Same list builds both an fp32mp2 array (F = LUT_FF) and a double reference
// array (F = LUT_ID).
#define LUT_ID(x) x
#define LUT_FF(x) ffloat(x)
#define LUT_LIST(F)                                                                                             \
  F(3.14159265358979323846), F(2.71828182845904523536), F(1.41421356237309504880), F(1.73205080756887729352),   \
    F(0.69314718055994530942), F(0.43429448190325182765), F(1.61803398874989484820), F(0.57721566490153286060), \
    F(299792458.0), F(6.62607015e-34), F(1.602176634e-19), F(9.10938356e-31), F(1.234567890123456789),          \
    F(9.876543210987654321), F(0.123456789012345678), F(123456.789012345678)

constexpr int LUT_SIZE = 16;

// Compile-time guard (mirrors the cuda_multi_fp float-float property): a double
// literal MUST decompose into (hi, lo) entirely at compile time. If fpmp2(double)
// ever stops being constexpr, or the split changes, these fail to COMPILE - so a
// regression can never silently reach runtime.
namespace
{
constexpr ffloat _ct_lut[] = {LUT_LIST(LUT_FF)};
constexpr double _ct_ref[] = {LUT_LIST(LUT_ID)};

// hi() is the nearest float; lo() is the exact residual - both evaluated by the
// constexpr ctor + constexpr accessors, i.e. purely at compile time.
static_assert(_ct_lut[0].hi() == (float) _ct_ref[0], "fp32mp2(double) hi must be a compile-time float cast (small)");
static_assert(_ct_lut[0].lo() == (float) (_ct_ref[0] - (double) (float) _ct_ref[0]),
              "fp32mp2(double) lo must be a compile-time residual (small)");
static_assert(_ct_lut[8].hi() == (float) _ct_ref[8], "fp32mp2(double) hi must be a compile-time float cast (large)");
static_assert(_ct_lut[8].lo() == (float) (_ct_ref[8] - (double) (float) _ct_ref[8]),
              "fp32mp2(double) lo must be a compile-time residual (large)");
// An exactly representable literal yields a zero residual at compile time.
static_assert(ffloat{1.5}.hi() == 1.5f && ffloat{1.5}.lo() == 0.0f,
              "exact double literal must decompose to (hi, lo == 0) at compile time");
} // namespace

// Multiply pairs of LUT entries and verify against the double reference; also
// check the round-trip precision of every stored entry.
TEST_HOST_DEVICE_FUNC void run_test()
{
  constexpr ffloat lut[] = {LUT_LIST(LUT_FF)};
  constexpr double ref[] = {LUT_LIST(LUT_ID)};

  const int idx[][2] = {{0, 0}, {0, 1}, {2, 3}, {4, 5}, {6, 7}, {12, 13}, {14, 15}, {0, 2}};
  const int num_ops  = (int) (sizeof(idx) / sizeof(idx[0]));

  const double tol = 1e-12;

  for (int i = 0; i < num_ops; i++)
  {
    const int a       = idx[i][0];
    const int b       = idx[i][1];
    const double prod = (double) (lut[a] * lut[b]);
    const double r    = ref[a] * ref[b];
    const double rel  = (r != 0.0) ? ::cuda::std::fabs(prod - r) / ::cuda::std::fabs(r) : 0.0;
    assert(rel < tol);
  }

  for (int i = 0; i < LUT_SIZE; i++)
  {
    const double rel = (ref[i] != 0.0) ? ::cuda::std::fabs((double) lut[i] - ref[i]) / ::cuda::std::fabs(ref[i]) : 0.0;
    assert(rel < tol);
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

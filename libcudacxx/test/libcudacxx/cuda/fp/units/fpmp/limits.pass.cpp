// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cuda::std::numeric_limits<fpmp2> specialization.
//
//  Validates the numeric_limits<> specialization for the double-word types
//  fp32mp2 (double-float) and fp64mp2 (double-double). Compile-time static_asserts
//  cover every reported characteristic (integer traits and the exact power-of-two
//  hi/lo components of min()/max()/lowest()/epsilon(), plus cv-qualified and
//  accuracy-variant forwarding). The run_test() then exercises the value members
//  at runtime.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

namespace cs = cuda::std;

template <class T>
using nl = cs::numeric_limits<T>;

//==========================================================================================
// Compile-time checks (evaluated for both host and device builds)
//==========================================================================================

// ----- fp32mp2 (double-float) -----
static_assert(nl<cudax::fp32mp2>::is_specialized, "fp32mp2 must be specialized");
static_assert(nl<cudax::fp32mp2>::is_signed, "fp32mp2 is signed");
static_assert(!nl<cudax::fp32mp2>::is_integer, "fp32mp2 is not integer");
static_assert(!nl<cudax::fp32mp2>::is_exact, "fp32mp2 is not exact");
static_assert(nl<cudax::fp32mp2>::radix == 2, "fp32mp2 radix is 2");
static_assert(nl<cudax::fp32mp2>::digits == 46, "fp32mp2 has 2*24-2 = 46 mantissa bits");
static_assert(nl<cudax::fp32mp2>::digits10 == 13, "fp32mp2 digits10");
static_assert(nl<cudax::fp32mp2>::max_digits10 == 15, "fp32mp2 max_digits10");
static_assert(nl<cudax::fp32mp2>::max_exponent == nl<float>::max_exponent, "fp32mp2 shares float's max exponent");
static_assert(nl<cudax::fp32mp2>::min_exponent == nl<float>::min_exponent + nl<float>::digits, "fp32mp2 min exponent");
static_assert(!nl<cudax::fp32mp2>::is_iec559, "double-word is not an IEEE-754 format");
static_assert(nl<cudax::fp32mp2>::is_bounded, "fp32mp2 is bounded");
static_assert(nl<cudax::fp32mp2>::has_infinity, "fp32mp2 has infinity");
static_assert(nl<cudax::fp32mp2>::has_quiet_NaN, "fp32mp2 has quiet NaN");
static_assert(nl<cudax::fp32mp2>::round_style == cs::round_to_nearest, "fp32mp2 rounds to nearest");

// exact (hi, lo) constants -- all powers of two, so equality is exact.
static_assert(nl<cudax::fp32mp2>::epsilon().hi() == 0x1p-45f, "fp32mp2 epsilon = 2^(1-46) = 2^-45");
static_assert(nl<cudax::fp32mp2>::epsilon().lo() == 0.0f, "fp32mp2 epsilon lo is zero");
static_assert(nl<cudax::fp32mp2>::max().hi() == nl<float>::max(), "fp32mp2 max hi = FLT_MAX");
static_assert(nl<cudax::fp32mp2>::max().lo() == nl<float>::max() * 0x1p-25f, "fp32mp2 max lo");
static_assert(nl<cudax::fp32mp2>::min().hi() == 0x1p-102f, "fp32mp2 min hi = 2^-102 (smallest all-normal)");
static_assert(nl<cudax::fp32mp2>::lowest().hi() == -nl<float>::max(), "fp32mp2 lowest = -max");
static_assert(nl<cudax::fp32mp2>::round_error().hi() == 0.5f, "fp32mp2 round_error = 0.5");

// ----- fp64mp2 (double-double) -----
static_assert(nl<cudax::fp64mp2>::is_specialized, "fp64mp2 must be specialized");
static_assert(nl<cudax::fp64mp2>::is_signed, "fp64mp2 is signed");
static_assert(nl<cudax::fp64mp2>::radix == 2, "fp64mp2 radix is 2");
static_assert(nl<cudax::fp64mp2>::digits == 104, "fp64mp2 has 2*53-2 = 104 mantissa bits");
static_assert(nl<cudax::fp64mp2>::digits10 == 31, "fp64mp2 digits10");
static_assert(nl<cudax::fp64mp2>::max_digits10 == 33, "fp64mp2 max_digits10");
static_assert(nl<cudax::fp64mp2>::max_exponent == nl<double>::max_exponent, "fp64mp2 shares double's max exponent");
static_assert(nl<cudax::fp64mp2>::min_exponent == nl<double>::min_exponent + nl<double>::digits,
              "fp64mp2 min exponent");
static_assert(nl<cudax::fp64mp2>::min_exponent == -968, "fp64mp2 min exponent matches __ibm128 (-968)");
static_assert(!nl<cudax::fp64mp2>::is_iec559, "double-word is not an IEEE-754 format");
static_assert(nl<cudax::fp64mp2>::has_infinity, "fp64mp2 has infinity");
static_assert(nl<cudax::fp64mp2>::has_quiet_NaN, "fp64mp2 has quiet NaN");

// exact (hi, lo) constants.
static_assert(nl<cudax::fp64mp2>::epsilon().hi() == 0x1p-103, "fp64mp2 epsilon = 2^(1-104) = 2^-103");
static_assert(nl<cudax::fp64mp2>::max().hi() == nl<double>::max(), "fp64mp2 max hi = DBL_MAX");
static_assert(nl<cudax::fp64mp2>::max().lo() == nl<double>::max() * 0x1p-54, "fp64mp2 max lo");
static_assert(nl<cudax::fp64mp2>::min().hi() == 0x1p-969, "fp64mp2 min hi = 2^-969 (matches __ibm128)");
static_assert(nl<cudax::fp64mp2>::lowest().hi() == -nl<double>::max(), "fp64mp2 lowest = -max");

// ----- accuracy variants and cv-qualified forwarding -----
static_assert(nl<cudax::fp32mp2_low>::digits == 46, "low variant is specialized");
static_assert(nl<cudax::fp32mp2_high>::is_specialized, "high variant is specialized");
static_assert(nl<cudax::fp64mp2_low>::digits == 104, "low variant is specialized");
static_assert(nl<cudax::fp64mp2_high>::max_exponent == nl<double>::max_exponent, "high variant is specialized");
static_assert(nl<const cudax::fp32mp2>::digits == 46, "const-qualified forwards to the specialization");
static_assert(nl<volatile cudax::fp64mp2>::digits == 104, "volatile-qualified forwards to the specialization");

// the value members are usable in a constexpr context (they use the constexpr (hi, lo) ctor).
static constexpr cudax::fp32mp2 kEps32 = nl<cudax::fp32mp2>::epsilon();
static constexpr cudax::fp64mp2 kMax64 = nl<cudax::fp64mp2>::max();
static_assert(kEps32.hi() > 0.0f && kMax64.hi() > 0.0, "constexpr value members");

//==========================================================================================
// Runtime checks (host + device)
//==========================================================================================

// Labeled per-check helper: returns the predicate so callers can accumulate.
TEST_HOST_DEVICE_FUNC inline bool fp_check(const char* label, bool pass)
{
  (void) label;
  return pass;
}

// (1 + epsilon) != 1, infinity() > max(), quiet_NaN() != quiet_NaN(), lowest() < 0.
// Each check is run unconditionally (no short-circuit) so every diagnostic is
// evaluated even when an earlier check fails; the results are combined at the end.
TEST_HOST_DEVICE_FUNC bool run_test()
{
  const cudax::fp32mp2 one32(1.0f);
  const bool c_eps32 = fp_check("fp32mp2: (1 + epsilon) != 1", (one32 + nl<cudax::fp32mp2>::epsilon()) != one32);

  const cudax::fp64mp2 one64(1.0);
  const bool c_eps64 = fp_check("fp64mp2: (1 + epsilon) != 1", (one64 + nl<cudax::fp64mp2>::epsilon()) != one64);

  const bool c_inf32 = fp_check(
    "fp32mp2: infinity() > max()", (double) nl<cudax::fp32mp2>::infinity() > (double) nl<cudax::fp32mp2>::max());

  const double nan64 = (double) nl<cudax::fp64mp2>::quiet_NaN();
  const bool c_nan64 = fp_check("fp64mp2: quiet_NaN() != quiet_NaN()", nan64 != nan64);

  const bool c_low64 = fp_check("fp64mp2: lowest() < 0", (double) nl<cudax::fp64mp2>::lowest() < 0.0);

  return c_eps32 && c_eps64 && c_inf32 && c_nan64 && c_low64;
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(run_test());
}

int main(int, char**)
{
  test();

  return 0;
}

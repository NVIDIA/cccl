// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cross-accuracy conversion semantics.
//
//  Different fpmp2<FpType, met> specializations share the same (hi, lo)
//  representation; only the accuracy tag differs. fpmp.h provides an explicit
//  cross-accuracy converting constructor that bit-copies (hi, lo). Compile-time
//  static_asserts pin the contract (explicit-only across accuracy, implicit
//  same-type, cross-FpType widening rules, and the assignment side). The
//  TEST_HOST_DEVICE_FUNC run_test() then confirms the conversions are bit-exact at
//  runtime.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// ---------------------------------------------------------------------------
// Compile-time contract for fp32mp2 (FpType == float).
// ---------------------------------------------------------------------------
// explicit construction across accuracy levels is allowed ...
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_low, cudax::fp32mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_low, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2, cudax::fp32mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_high, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_high, cudax::fp32mp2_low>::value, "");

// ... but implicit conversion across accuracy levels is NOT.
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_high, cudax::fp32mp2_low>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2, cudax::fp32mp2_low>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp32mp2>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_high, cudax::fp32mp2>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2, cudax::fp32mp2_high>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp32mp2_high>::value, "");

// Same-type implicit conversion (copy) is unaffected.
static_assert(::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2_high, cudax::fp32mp2_high>::value, "");

// Cross-FpType conversion contract: upconvert implicit, downconvert honors the
// CCCL_FPMP_EXPLICIT_CASTS knob. Both hold only while the accuracy tag is
// preserved (fpmp2_accuracy::def aliases mid, so fp32mp2 -> fp64mp2 counts as
// tag-preserving).
static_assert(::cuda::std::is_convertible<cudax::fp32mp2, cudax::fp64mp2>::value,
              "fp32mp2 -> fp64mp2 must be implicit (lossless upconvert)");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp64mp2_low>::value, "");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2_high, cudax::fp64mp2_high>::value, "");
#if CCCL_FPMP_EXPLICIT_CASTS == 1
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2, cudax::fp32mp2>::value,
              "fp64mp2 -> fp32mp2 must be explicit under EXPLICIT_CASTS=1");
#else
static_assert(::cuda::std::is_convertible<cudax::fp64mp2, cudax::fp32mp2>::value,
              "fp64mp2 -> fp32mp2 implicit by default (matches double -> fp32mp2)");
#endif

// Switching the accuracy tag stays explicit-only when the precision changes too,
// so the rule above holds for every conversion into fpmp2, not just the
// same-precision one. cross_prec.pass.cpp pins the full matrix.
static_assert(::cuda::std::is_constructible<cudax::fp64mp2_high, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_low, cudax::fp64mp2_high>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp64mp2_high>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2_high, cudax::fp32mp2_low>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp64mp2_high&, cudax::fp32mp2_low>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_low&, cudax::fp64mp2_high>::value, "");

// Assignment side of the contract: cross-accuracy assignment must fail (the
// explicit ctor is not visible to copy-assignment); same-type assignment works.
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_low&, cudax::fp32mp2_high>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_low&, cudax::fp32mp2>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2&, cudax::fp32mp2_low>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2&, cudax::fp32mp2_high>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_high&, cudax::fp32mp2>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_high&, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp32mp2_low&, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp32mp2&, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp32mp2_high&, cudax::fp32mp2_high>::value, "");

// The new ctor preserves result type (no type inference surprises).
static_assert(::cuda::std::is_same<decltype(cudax::fp32mp2_low(::cuda::std::declval<cudax::fp32mp2_high>())),
                                   cudax::fp32mp2_low>::value,
              "");

// ---------------------------------------------------------------------------
// Compile-time contract for fp64mp2 (FpType == double).
// ---------------------------------------------------------------------------
static_assert(::cuda::std::is_constructible<cudax::fp64mp2_low, cudax::fp64mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp64mp2_high, cudax::fp64mp2>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2_high, cudax::fp64mp2_low>::value, "");
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2, cudax::fp64mp2_low>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp64mp2_low&, cudax::fp64mp2_high>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp64mp2_low&, cudax::fp64mp2>::value, "");

// ---------------------------------------------------------------------------
// Runtime bit-equality check.
// ---------------------------------------------------------------------------
template <typename Dst, typename Src>
TEST_HOST_DEVICE_FUNC static bool bit_exact(Src src)
{
  Dst dst(src); // explicit cross-accuracy ctor
  return (dst.hi() == src.hi()) && (dst.lo() == src.lo());
}

TEST_HOST_DEVICE_FUNC void run_test()
{
  // Representative (hi, lo) pairs: regular, near-max, tiny, negative.
  const float f32[4][2] = {
    {1.2345678f, 1.0e-9f},
    {0x1.fffffep+126f, -0x1.0p+102f},
    {1.0e-30f, 1.0e-38f},
    {-3.1415927f, 1.5e-8f},
  };
  for (int i = 0; i < 4; ++i)
  {
    cudax::fp32mp2 sd(f32[i][0], f32[i][1]);
    assert(bit_exact<cudax::fp32mp2_low>(sd) && bit_exact<cudax::fp32mp2_high>(sd));
    cudax::fp32mp2_low sl(f32[i][0], f32[i][1]);
    assert(bit_exact<cudax::fp32mp2>(sl) && bit_exact<cudax::fp32mp2_high>(sl));
    cudax::fp32mp2_high sh(f32[i][0], f32[i][1]);
    assert(bit_exact<cudax::fp32mp2>(sh) && bit_exact<cudax::fp32mp2_low>(sh));
  }

  const double f64[4][2] = {
    {1.234567890123456, 1.0e-18},
    {1.0e+300, 1.0e+283},
    {-2.7182818284590452, 1.5e-17},
    {1.0e-200, 1.0e-217},
  };
  for (int i = 0; i < 4; ++i)
  {
    cudax::fp64mp2_high sh(f64[i][0], f64[i][1]);
    assert(bit_exact<cudax::fp64mp2_low>(sh));
    cudax::fp64mp2 sd(f64[i][0], f64[i][1]);
    assert(bit_exact<cudax::fp64mp2_low>(sd));
  }

  // Every explicit-conversion shape routes through the bit-exact ctor.
  {
    cudax::fp32mp2_high src(0x1.23p+4f, 0x1.0p-20f);
    cudax::fp32mp2_low a(src); // direct-init
    cudax::fp32mp2_low b = cudax::fp32mp2_low(src); // functional cast in copy-init
    cudax::fp32mp2_low c = static_cast<cudax::fp32mp2_low>(src); // static_cast in copy-init
    cudax::fp32mp2_low d;
    d = cudax::fp32mp2_low(src); // explicit assign (functional)
    cudax::fp32mp2_low e;
    e = static_cast<cudax::fp32mp2_low>(src); // explicit assign (static_cast)
    assert((a.hi() == src.hi() && a.lo() == src.lo()) && (b.hi() == src.hi() && b.lo() == src.lo())
           && (c.hi() == src.hi() && c.lo() == src.lo()) && (d.hi() == src.hi() && d.lo() == src.lo())
           && (e.hi() == src.hi() && e.lo() == src.lo()));
  }

  // Changing precision and accuracy at once must give the same pair as changing
  // the tag first (bit copy) and then the precision.
  {
    cudax::fp32mp2_low src{1.2345678f, 0x1.0p-30f};
    cudax::fp64mp2_high wide{src};
    cudax::fp64mp2_high wide_ref{cudax::fp32mp2_high{src}};
    assert(wide.hi() == wide_ref.hi() && wide.lo() == wide_ref.lo());

    cudax::fp64mp2_low src64{1.234567890123456, 1.0e-18};
    cudax::fp32mp2_high narrow{src64};
    cudax::fp32mp2_high narrow_ref{cudax::fp64mp2_high{src64}};
    assert(narrow.hi() == narrow_ref.hi() && narrow.lo() == narrow_ref.lo());
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

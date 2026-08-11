// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cross-precision fp32mp2 <-> fp64mp2 conversion.
//
//  Verifies the cross-precision converting constructors / assignment operators:
//    - Upconvert fp32mp2 -> fp64mp2 is implicit and lossless (fast_two_sum on the
//      float->double promotions preserves the residual even for wide-exponent-gap
//      inputs like (1.0f, 2^-100f) that are not representable as a single double).
//    - Downconvert fp64mp2 -> fp32mp2 honors CCCL_FPMP_EXPLICIT_CASTS, is bounded
//      to ~2 ulp of fp32mp2 precision, and is bit-exact for fp32mp2-born values.
//    - A precision change that also switches the accuracy tag is explicit in both
//      directions, and has no assignment form.
//    - Round trip fp32mp2 -> fp64mp2 -> fp32mp2 is bit-exact.
//    - Every explicit-conversion form and the assignment overload agree.
//  The convertibility/assignability matrix is pinned by static_asserts. The same
//  TEST_HOST_DEVICE_FUNC run_test() runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// ---------------------------------------------------------------------------
// Compile-time contract.
// ---------------------------------------------------------------------------
// Upconvert (fp32mp2 -> fp64mp2): implicit and lossless while the accuracy tag
// is preserved. Note that fpmp2_accuracy::def aliases mid, so fp32mp2 -> fp64mp2
// is tag-preserving.
static_assert(::cuda::std::is_constructible<cudax::fp64mp2, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp64mp2, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp64mp2, cudax::fp32mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp64mp2_low, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp64mp2_high, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2, cudax::fp64mp2>::value,
              "fp32mp2 -> fp64mp2 must be implicit");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp64mp2_low>::value, "");
static_assert(::cuda::std::is_convertible<cudax::fp32mp2_high, cudax::fp64mp2_high>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp64mp2&, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp64mp2_low&, cudax::fp32mp2_low>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp64mp2_high&, cudax::fp32mp2_high>::value, "");
static_assert(::cuda::std::is_same<decltype(cudax::fp64mp2_low(::cuda::std::declval<cudax::fp32mp2_high>())),
                                   cudax::fp64mp2_low>::value,
              "");

// A widening that also switches the accuracy tag is explicit-only: the tag picks
// the arithmetic algorithm, so it must be opt-in just like the same-precision
// cross-accuracy conversion. Assignment is not offered across tags at all, which
// leaves an explicit cast as the only spelling.
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp64mp2_high>::value,
              "cross-accuracy upconvert must NOT be implicit");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_high, cudax::fp64mp2_low>::value,
              "cross-accuracy upconvert must NOT be implicit");
static_assert(!::cuda::std::is_convertible<cudax::fp32mp2_low, cudax::fp64mp2>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp64mp2_low&, cudax::fp32mp2_high>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp64mp2_high&, cudax::fp32mp2_low>::value, "");

// Downconvert (fp64mp2 -> fp32mp2): explicit-macro-driven when the accuracy tag
// is preserved, always explicit when it changes.
static_assert(::cuda::std::is_constructible<cudax::fp32mp2, cudax::fp64mp2>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2, cudax::fp64mp2_low>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2, cudax::fp64mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_low, cudax::fp64mp2>::value, "");
static_assert(::cuda::std::is_constructible<cudax::fp32mp2_high, cudax::fp64mp2>::value, "");
#if CCCL_FPMP_EXPLICIT_CASTS == 1
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2, cudax::fp32mp2>::value,
              "downconvert must NOT be implicit under EXPLICIT_CASTS=1");
#else
static_assert(::cuda::std::is_convertible<cudax::fp64mp2, cudax::fp32mp2>::value,
              "downconvert is implicit under EXPLICIT_CASTS=0");
#endif
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2_low, cudax::fp32mp2_high>::value,
              "cross-accuracy downconvert must NOT be implicit");
static_assert(!::cuda::std::is_convertible<cudax::fp64mp2_high, cudax::fp32mp2>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp32mp2&, cudax::fp64mp2>::value, "");
static_assert(::cuda::std::is_assignable<cudax::fp32mp2_low&, cudax::fp64mp2_low>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_low&, cudax::fp64mp2_high>::value, "");
static_assert(!::cuda::std::is_assignable<cudax::fp32mp2_high&, cudax::fp64mp2_low>::value, "");
static_assert(::cuda::std::is_same<decltype(cudax::fp32mp2_high(::cuda::std::declval<cudax::fp64mp2_low>())),
                                   cudax::fp32mp2_high>::value,
              "");

// ---------------------------------------------------------------------------
// Runtime checks.
// ---------------------------------------------------------------------------
struct F32
{
  float hi, lo;
};
struct F64
{
  double hi, lo;
};

// Upconvert must be bit-exact vs an inlined fast_two_sum reference and produce a
// renormalized (|lo| <= ulp(hi)/2) fp64mp2 pair.
TEST_HOST_DEVICE_FUNC static void check_upconvert(const F32& src, const F64& dst)
{
  const double a      = (double) src.hi;
  const double b      = (double) src.lo;
  const double ref_hi = a + b;
  const double z      = ref_hi - a;
  const double ref_lo = b - z;

  assert(dst.hi == ref_hi);
  assert(dst.lo == ref_lo);

  if (dst.hi == 0.0)
  {
    assert(dst.lo == 0.0);
  }
  else
  {
    const double ulp_hi = ::cuda::std::ldexp(1.0, ::cuda::std::ilogb(dst.hi) - 52);
    assert(::cuda::std::fabs(dst.lo) <= 0.5 * ulp_hi);
  }
}

TEST_HOST_DEVICE_FUNC static void check_downconvert(const F64& src, const F32& dst, double tol)
{
  const double ref     = (double) src.hi + (double) src.lo;
  const double got     = (double) dst.hi + (double) dst.lo;
  const double abs_ref = ::cuda::std::fabs(ref);
  const double rel_err = (abs_ref > 0.0) ? ::cuda::std::fabs((got - ref) / ref) : ::cuda::std::fabs(got - ref);
  assert(rel_err <= tol);
}

TEST_HOST_DEVICE_FUNC void run_test()
{
  // Upconvert inputs spanning the renormalization spectrum.
  const F32 in32[8] = {
    {1.2345678f, 1.0e-9f}, // ordinary
    {1.0f, 0x1.0p-100f}, // pathological: lo far below ulp(hi)
    {-3.1415927f, 1.5e-8f}, // negative hi
    {1.0e+30f, 0x1.0p-110f}, // wide exponent gap
    {0x1.0p+120f, 0x1.0p-149f}, // near float max
    {0.0f, 0.0f}, // zero
    {3.14f, 0.0f}, // hi only
    {1.0f, 0x1.0p-25f}, // |lo| ~ ulp(hi)/2
  };
  for (int i = 0; i < 8; ++i)
  {
    cudax::fp32mp2 src(in32[i].hi, in32[i].lo);
    cudax::fp64mp2 dst = src; // implicit upconvert
    F64 o              = {dst.hi(), dst.lo()};
    check_upconvert(in32[i], o);
  }

  // Residual must survive in dst.lo for the non-single-double inputs (1, 3, 4).
  for (int i : {1, 3, 4})
  {
    cudax::fp32mp2 src(in32[i].hi, in32[i].lo);
    cudax::fp64mp2 dst = src;
    assert(dst.lo() != 0.0);
  }

  // Downconvert inputs (values needing > 24 bits).
  const F64 in64[4] = {
    {1.234567890123456, 1.0e-18}, // canonical double
    {1.0e+30, 1.5e+13}, // large magnitude
    {-2.7182818284590452, 1.5e-17}, // negative
    {(double) 1.2345678f, (double) 1.0e-9f}, // fp32mp2-born (must be exact)
  };
  const double tols[4] = {1.5e-14, 1.5e-14, 1.5e-14, 0.0};
  for (int i = 0; i < 4; ++i)
  {
    cudax::fp64mp2 src(in64[i].hi, in64[i].lo);
    cudax::fp32mp2 dst(src); // explicit downconvert (direct-init)
    F32 o = {dst.hi(), dst.lo()};
    check_downconvert(in64[i], o, tols[i]);
  }

  // Round trip fp32mp2 -> fp64mp2 -> fp32mp2 must be bit-exact.
  const F32 rt[3] = {
    {1.0f, 0x1.0p-100f}, // pathological
    {1.2345678f, 1.0e-9f}, // ordinary
    {0x1.0p+120f, 0x1.0p-149f}, // near float max
  };
  for (int i = 0; i < 3; ++i)
  {
    cudax::fp32mp2 a(rt[i].hi, rt[i].lo);
    cudax::fp64mp2 b = a; // implicit upconvert
    cudax::fp32mp2 c = static_cast<cudax::fp32mp2>(b); // explicit downconvert
    assert((c.hi() == rt[i].hi) && (c.lo() == rt[i].lo));
  }

  // Every explicit conversion form (and operator=) must produce the same pair.
  {
    cudax::fp64mp2 src(1.234567890123456, 1.0e-18);
    cudax::fp32mp2 a(src); // direct-init
    cudax::fp32mp2 b = cudax::fp32mp2(src); // functional cast
    cudax::fp32mp2 c = static_cast<cudax::fp32mp2>(src); // static_cast
    cudax::fp32mp2 d;
    d = static_cast<cudax::fp32mp2>(src); // assign via cast
    cudax::fp32mp2 e;
    e = src; // operator= overload
    assert((a.hi() == b.hi() && a.lo() == b.lo()) && (a.hi() == c.hi() && a.lo() == c.lo())
           && (a.hi() == d.hi() && a.lo() == d.lo()) && (a.hi() == e.hi() && a.lo() == e.lo()));
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

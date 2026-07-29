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
//    - Round trip fp32mp2 -> fp64mp2 -> fp32mp2 is bit-exact.
//    - Every explicit-conversion form and the assignment overload agree.
//  The convertibility/assignability matrix is pinned by static_asserts. The same
//  _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// ---------------------------------------------------------------------------
// Compile-time contract.
// ---------------------------------------------------------------------------
// Upconvert (fp32mp2 -> fp64mp2): implicit, lossless.
static_assert(::cuda::std::is_constructible<fp64mp2, fp32mp2>::value, "");
static_assert(::cuda::std::is_constructible<fp64mp2, fp32mp2_low>::value, "");
static_assert(::cuda::std::is_constructible<fp64mp2, fp32mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<fp64mp2_low, fp32mp2>::value, "");
static_assert(::cuda::std::is_constructible<fp64mp2_high, fp32mp2>::value, "");
static_assert(::cuda::std::is_convertible<fp32mp2, fp64mp2>::value, "fp32mp2 -> fp64mp2 must be implicit");
static_assert(::cuda::std::is_convertible<fp32mp2_low, fp64mp2_high>::value,
              "cross-accuracy upconvert must be implicit");
static_assert(::cuda::std::is_convertible<fp32mp2_high, fp64mp2_low>::value,
              "cross-accuracy upconvert must be implicit");
static_assert(::cuda::std::is_assignable<fp64mp2&, fp32mp2>::value, "");
static_assert(::cuda::std::is_assignable<fp64mp2_low&, fp32mp2_high>::value, "");
static_assert(::cuda::std::is_assignable<fp64mp2_high&, fp32mp2_low>::value, "");
static_assert(::cuda::std::is_same<decltype(fp64mp2_low(::cuda::std::declval<fp32mp2_high>())), fp64mp2_low>::value,
              "");

// Downconvert (fp64mp2 -> fp32mp2): explicit-macro-driven.
static_assert(::cuda::std::is_constructible<fp32mp2, fp64mp2>::value, "");
static_assert(::cuda::std::is_constructible<fp32mp2, fp64mp2_low>::value, "");
static_assert(::cuda::std::is_constructible<fp32mp2, fp64mp2_high>::value, "");
static_assert(::cuda::std::is_constructible<fp32mp2_low, fp64mp2>::value, "");
static_assert(::cuda::std::is_constructible<fp32mp2_high, fp64mp2>::value, "");
#if CCCL_FPMP_EXPLICIT_CASTS == 1
static_assert(!::cuda::std::is_convertible<fp64mp2, fp32mp2>::value,
              "downconvert must NOT be implicit under EXPLICIT_CASTS=1");
#else
static_assert(::cuda::std::is_convertible<fp64mp2, fp32mp2>::value, "downconvert is implicit under EXPLICIT_CASTS=0");
#endif
static_assert(::cuda::std::is_assignable<fp32mp2&, fp64mp2>::value, "");
static_assert(::cuda::std::is_assignable<fp32mp2_low&, fp64mp2_high>::value, "");
static_assert(::cuda::std::is_assignable<fp32mp2_high&, fp64mp2_low>::value, "");
static_assert(::cuda::std::is_same<decltype(fp32mp2_high(::cuda::std::declval<fp64mp2_low>())), fp32mp2_high>::value,
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
_CCCL_HOST_DEVICE static bool check_upconvert(const F32& src, const F64& dst)
{
  const double a      = (double) src.hi;
  const double b      = (double) src.lo;
  const double ref_hi = a + b;
  const double z      = ref_hi - a;
  const double ref_lo = b - z;

  const bool bit_exact = (dst.hi == ref_hi) && (dst.lo == ref_lo);

  bool renorm;
  if (dst.hi == 0.0)
  {
    renorm = (dst.lo == 0.0);
  }
  else
  {
    const double ulp_hi = ::cuda::std::ldexp(1.0, ::cuda::std::ilogb(dst.hi) - 52);
    renorm              = ::cuda::std::fabs(dst.lo) <= 0.5 * ulp_hi;
  }
  return bit_exact && renorm;
}

_CCCL_HOST_DEVICE static bool check_downconvert(const F64& src, const F32& dst, double tol)
{
  const double ref     = (double) src.hi + (double) src.lo;
  const double got     = (double) dst.hi + (double) dst.lo;
  const double abs_ref = ::cuda::std::fabs(ref);
  const double rel_err = (abs_ref > 0.0) ? ::cuda::std::fabs((got - ref) / ref) : ::cuda::std::fabs(got - ref);
  return rel_err <= tol;
}

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok = true;

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
    fp32mp2 src(in32[i].hi, in32[i].lo);
    fp64mp2 dst = src; // implicit upconvert
    F64 o       = {dst.hi(), dst.lo()};
    ok          = ok && check_upconvert(in32[i], o);
  }

  // Residual must survive in dst.lo for the non-single-double inputs (1, 3, 4).
  for (int i : {1, 3, 4})
  {
    fp32mp2 src(in32[i].hi, in32[i].lo);
    fp64mp2 dst = src;
    ok          = ok && (dst.lo() != 0.0);
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
    fp64mp2 src(in64[i].hi, in64[i].lo);
    fp32mp2 dst(src); // explicit downconvert (direct-init)
    F32 o = {dst.hi(), dst.lo()};
    ok    = ok && check_downconvert(in64[i], o, tols[i]);
  }

  // Round trip fp32mp2 -> fp64mp2 -> fp32mp2 must be bit-exact.
  const F32 rt[3] = {
    {1.0f, 0x1.0p-100f}, // pathological
    {1.2345678f, 1.0e-9f}, // ordinary
    {0x1.0p+120f, 0x1.0p-149f}, // near float max
  };
  for (int i = 0; i < 3; ++i)
  {
    fp32mp2 a(rt[i].hi, rt[i].lo);
    fp64mp2 b = a; // implicit upconvert
    fp32mp2 c = static_cast<fp32mp2>(b); // explicit downconvert
    ok        = ok && (c.hi() == rt[i].hi) && (c.lo() == rt[i].lo);
  }

  // Every explicit conversion form (and operator=) must produce the same pair.
  {
    fp64mp2 src(1.234567890123456, 1.0e-18);
    fp32mp2 a(src); // direct-init
    fp32mp2 b = fp32mp2(src); // functional cast
    fp32mp2 c = static_cast<fp32mp2>(src); // static_cast
    fp32mp2 d;
    d = static_cast<fp32mp2>(src); // assign via cast
    fp32mp2 e;
    e  = src; // operator= overload
    ok = ok && (a.hi() == b.hi() && a.lo() == b.lo()) && (a.hi() == c.hi() && a.lo() == c.lo())
      && (a.hi() == d.hi() && a.lo() == d.lo()) && (a.hi() == e.hi() && a.lo() == e.lo());
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

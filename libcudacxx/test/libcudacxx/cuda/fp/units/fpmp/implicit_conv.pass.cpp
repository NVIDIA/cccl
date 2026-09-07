// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: no implicit conversion into fpmp2 may lose the low limb.
//
//  A narrowing source must not reach an fpmp2 implicitly, because the only
//  constructor it could reach that way is the single-limb one, which zeroes lo
//  and reports nothing. Under CCCL_FPMP_EXPLICIT_CASTS=1 such a source is
//  therefore not convertible at all, only constructible, so the cast has to be
//  written out. Under =0 it converts implicitly, but still through the accurate
//  two-limb path, so the result is identical to direct-initialization either way.
//
//  A source the pair represents exactly for every value of its type is not
//  narrowing and stays implicit under both settings. That covers int32_t and
//  narrower into either width, and every standard integer into fp64mp2, so the
//  ordinary `fp64mp2 acc = 0;` spelling keeps working.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using ffloat  = cudax::fp32mp2;
using ddouble = cudax::fp64mp2;

// ============================ conversions ============================

// Every source stays constructible whatever the knob does: making a conversion
// explicit must remove it from copy-initialization, not from the type.
static_assert(::cuda::std::is_constructible_v<ffloat, double>);
static_assert(::cuda::std::is_constructible_v<ffloat, long long>);
static_assert(::cuda::std::is_constructible_v<ffloat, unsigned long long>);
static_assert(::cuda::std::is_constructible_v<ddouble, double>);

// Exact for every value of the source type, so implicit under both settings.
static_assert(::cuda::std::is_convertible_v<float, ffloat>);
static_assert(::cuda::std::is_convertible_v<float, ddouble>);
static_assert(::cuda::std::is_convertible_v<double, ddouble>);
static_assert(::cuda::std::is_convertible_v<bool, ffloat>);
static_assert(::cuda::std::is_convertible_v<char, ffloat>);
static_assert(::cuda::std::is_convertible_v<short, ffloat>);
static_assert(::cuda::std::is_convertible_v<::cuda::std::int32_t, ffloat>);
static_assert(::cuda::std::is_convertible_v<::cuda::std::uint32_t, ffloat>);
static_assert(::cuda::std::is_convertible_v<long long, ddouble>);
static_assert(::cuda::std::is_convertible_v<unsigned long long, ddouble>);

// Narrowing: needs bits the pair may not have, so the knob decides.
#if CCCL_FPMP_EXPLICIT_CASTS == 1
static_assert(!::cuda::std::is_convertible_v<double, ffloat>,
              "double -> fp32mp2 must be explicit under EXPLICIT_CASTS=1: the implicit path would drop lo");
static_assert(!::cuda::std::is_convertible_v<long long, ffloat>,
              "int64 -> fp32mp2 must be explicit under EXPLICIT_CASTS=1: 63 bits exceed the float pair");
static_assert(!::cuda::std::is_convertible_v<unsigned long long, ffloat>);
#else
static_assert(::cuda::std::is_convertible_v<double, ffloat>);
static_assert(::cuda::std::is_convertible_v<long long, ffloat>);
static_assert(::cuda::std::is_convertible_v<unsigned long long, ffloat>);
#endif

// ==================== scalar compound assignment =====================

// operator+=(scalar) is gated the same way: a bare component-type parameter would
// beat operator+=(const fpmp2&) for any narrowing source by standard conversion,
// then accumulate a value whose low limb was already gone.
template <class _Tp, class _Up, class = void>
inline constexpr bool has_plus_equal_v = false;
template <class _Tp, class _Up>
inline constexpr bool
  has_plus_equal_v<_Tp, _Up, decltype(void(::cuda::std::declval<_Tp&>() += ::cuda::std::declval<_Up>()))> = true;

static_assert(has_plus_equal_v<ffloat, float>);
static_assert(has_plus_equal_v<ffloat, int>);
static_assert(has_plus_equal_v<ddouble, double>);
#if CCCL_FPMP_EXPLICIT_CASTS == 1
static_assert(!has_plus_equal_v<ffloat, double>,
              "fp32mp2 += double must not compile under EXPLICIT_CASTS=1: it would truncate to float first");
#else
static_assert(has_plus_equal_v<ffloat, double>);
#endif

// ========================= compile-time value ========================

// The written-out cast is a constant expression and keeps both limbs, so a
// coefficient table can hold full-precision values without a runtime pass.
constexpr ffloat kCast = static_cast<ffloat>(1.2345678901234567);
static_assert(kCast.hi() == 1.23456788f, "hi limb must be folded at compile time");
static_assert(kCast.lo() != 0.0f, "low limb must survive the compile-time conversion");

#if CCCL_FPMP_EXPLICIT_CASTS == 0
// When the knob lets it through, the implicit form must land on the same value:
// permitting the conversion must not also mean taking the lossy path.
constexpr ffloat kImplicit = 1.2345678901234567;
static_assert(kImplicit.hi() == kCast.hi() && kImplicit.lo() == kCast.lo(),
              "implicit conversion must use the same two-limb path as the cast");
#endif

// ============================== runtime ==============================

// Whatever the knob, an allowed conversion must produce what direct-init produces.
// The integer constructors are not constexpr, so these run.
TEST_HOST_DEVICE_FUNC void run_test()
{
  // int32 does not fit one float but does fit a float pair, so lo must carry the
  // remainder rather than being zeroed.
  const ::cuda::std::int32_t i32 = (1 << 24) + 1; // first int32 a float cannot hold
  const ffloat i32_direct{i32};
  const ffloat i32_implicit = i32;
  assert(i32_direct.hi() == i32_implicit.hi());
  assert(i32_direct.lo() == i32_implicit.lo());
  assert(i32_implicit.lo() != 0.0f);
  assert(static_cast<double>(i32_implicit.hi()) + static_cast<double>(i32_implicit.lo()) == 16777217.0);

  // Same story one width up: int64 beyond 2^53 needs the low limb of a double pair.
  const long long i64 = (1LL << 53) + 1;
  const ddouble i64_direct{i64};
  const ddouble i64_implicit = i64;
  assert(i64_direct.hi() == i64_implicit.hi());
  assert(i64_direct.lo() == i64_implicit.lo());
  assert(i64_implicit.lo() != 0.0);
  assert(i64_implicit.hi() + i64_implicit.lo() == 9007199254740993.0);

  // An integer that one limb already holds exactly must take the single-component path,
  // leaving lo untouched at zero rather than computing a residual that is always zero.
  // int32_t fits a double limb, so this is the shape of `fp64mp2 acc = 0;` in a hot loop.
  const ddouble i32_wide = i32;
  assert(i32_wide.hi() == 16777217.0);
  assert(i32_wide.lo() == 0.0);

  // Small integers must still be plainly usable as an accumulator seed.
  ddouble acc = 0;
  assert(acc.hi() == 0.0 && acc.lo() == 0.0);
  acc += 1;
  assert(acc.hi() == 1.0);

  // The explicit cast keeps both limbs at runtime too, matching the constexpr result.
  const double d       = 1.2345678901234567;
  const ffloat as_pair = static_cast<ffloat>(d);
  assert(as_pair.hi() == kCast.hi());
  assert(as_pair.lo() == kCast.lo());
  assert(as_pair.lo() != 0.0f);
}

int main(int, char**)
{
  run_test();
  NV_IF_TARGET(NV_IS_HOST, (run_test();))
  return 0;
}

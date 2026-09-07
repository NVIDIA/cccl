// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: integer conversions with fp32mp2 (float-float).
//
//  Verifies integer-conversion correctness for fp32mp2:
//    - int32_t / uint32_t <-> fp32mp2 round-trips are exact (incl. values around
//      the 2^24 float-precision limit and INT32/UINT32 extremes), with sign
//      consistency (lo has the sign of the input) / non-negativity for unsigned.
//    - All standard integer types (short/int/long/long long + unsigned) construct
//      and convert back, and same-width types agree with the fixed-width path.
//    - fp32mp2 -> int truncates toward zero, summing hi+lo first (so values like
//      19.99999991... -> {20.0, -8.77e-8} truncate to 19, not 20).
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using ffloat = cudax::fp32mp2;

#if _CCCL_HAS_INT128()
// 128-bit integer construction and conversion are deliberately deleted: they would
// silently truncate to 64 bits. Verify fp32mp2 neither constructs from nor converts
// to __int128 while the standard integer widths remain usable.
static_assert(!::cuda::std::is_constructible_v<ffloat, __int128_t>, "");
static_assert(!::cuda::std::is_constructible_v<ffloat, __uint128_t>, "");
static_assert(!::cuda::std::is_constructible_v<__int128_t, ffloat>, "");
static_assert(!::cuda::std::is_constructible_v<__uint128_t, ffloat>, "");
static_assert(::cuda::std::is_constructible_v<ffloat, int64_t>, "");
static_assert(::cuda::std::is_constructible_v<int64_t, ffloat>, "");
#endif // _CCCL_HAS_INT128()

#if _CCCL_FPMP_FP128_ENABLE == 1
// Unlike the single-double emulated types (fpemu), the double-double fp64mp2 has
// enough mantissa (~106 bits) to hold a 128-bit float, so quad construction and
// conversion are deliberately SUPPORTED (not deleted). __fpmp_fp128 is __float128
// wherever _CCCL_HAS_FLOAT128() and long double on IEEE-128 long double platforms.
static_assert(::cuda::std::is_constructible_v<cudax::fp64mp2, cudax::__fpmp_fp128>, "");
static_assert(::cuda::std::is_constructible_v<cudax::__fpmp_fp128, cudax::fp64mp2>, "");

// fp32mp2 carries ~48 bits, fewer than a double, so quad is not its interchange
// type: both directions are deliberately deleted, like the 128-bit integers above.
// Without the deletions the constructor would report an ambiguity and the
// conversion would silently route through operator double().
static_assert(!::cuda::std::is_constructible_v<ffloat, cudax::__fpmp_fp128>, "");
static_assert(!::cuda::std::is_constructible_v<cudax::__fpmp_fp128, ffloat>, "");
// The double image stays reachable, spelled out.
static_assert(::cuda::std::is_constructible_v<cudax::__fpmp_fp128, double>, "");
#endif // _CCCL_FPMP_FP128_ENABLE == 1

TEST_HOST_DEVICE_FUNC void run_test()
{
  using ::cuda::std::int32_t;
  using ::cuda::std::int64_t;
  using ::cuda::std::uint32_t;

  // int32_t round-trip: exact, and lo carries the sign of the input. INT32_MIN is spelled
  // -2147483647 - 1 because -2147483648 is unary minus on a literal too large for int,
  // which MSVC types as unsigned long and then rejects as a narrowing conversion.
  {
    const int32_t vals[] = {
      0, 1, -1, 42, -42, 16777215, 16777216, 16777217, -16777215, -16777216, -16777217, 2147483647, -2147483647 - 1};
    for (int32_t v : vals)
    {
      ffloat x(v);
      const int32_t back = static_cast<int32_t>(x);
      assert(back == v);
      if (v > 0)
      {
        assert(x.lo() >= 0);
      }
      else if (v < 0)
      {
        assert(x.lo() <= 0);
      }
    }
  }

  // uint32_t round-trip: exact, both components non-negative.
  {
    const uint32_t vals[] = {0u, 1u, 42u, 16777215u, 16777216u, 16777217u, 4294967295u};
    for (uint32_t v : vals)
    {
      ffloat x(v);
      const uint32_t back = static_cast<uint32_t>(x);
      assert((back == v) && (x.hi() >= 0) && (x.lo() >= 0));
    }
  }

  // All standard integer types construct and convert back; long / long long
  // must agree with the fixed-width int64_t path.
  {
    const long long vals[] = {0, 1, -1, 42, -42, 65535, -65536, 1048576, -1048576};
    for (long long v : vals)
    {
      const short s      = static_cast<short>(v % 30000);
      const int i        = static_cast<int>(v);
      const long l       = static_cast<long>(v);
      const long long l2 = v;

      assert(static_cast<short>(ffloat(s)) == s && static_cast<int>(ffloat(i)) == i && static_cast<long>(ffloat(l)) == l
             && static_cast<long long>(ffloat(l2)) == l2
             && static_cast<long long>(ffloat(l2)) == static_cast<int64_t>(ffloat(static_cast<int64_t>(v))));

      if (v >= 0)
      {
        const unsigned int ui        = static_cast<unsigned int>(v);
        const unsigned long ul       = static_cast<unsigned long>(v);
        const unsigned long long ull = static_cast<unsigned long long>(v);
        assert(static_cast<unsigned int>(ffloat(ui)) == ui && static_cast<unsigned long>(ffloat(ul)) == ul
               && static_cast<unsigned long long>(ffloat(ull)) == ull);
      }
    }
  }

  // Truncation toward zero.
  {
    const double vals[]    = {2.7, 2.3, -2.7, -2.3, 0.9, -0.9};
    const int32_t expect[] = {2, 2, -2, -2, 0, 0};
    for (int i = 0; i < 6; ++i)
    {
      ffloat x(vals[i]);
      assert(static_cast<int32_t>(x) == expect[i]);
    }
  }

  // Negative low part: hi+lo must be summed before truncation.
  {
    const double vals[] = {
      1.9999999123341809E+01,
      9.999999523162842E+00,
      1.9999998807907104E+00,
      9.9999999E+01,
      1.23E+02,
      1.234567890123E+02,
      -1.9999999123341809E+01,
      -9.999999523162842E+00,
      2.9999999E+00,
      -2.9999999E+00,
    };
    const int32_t expect[] = {19, 9, 1, 99, 123, 123, -19, -9, 2, -2};
    for (int i = 0; i < 10; ++i)
    {
      ffloat x(vals[i]);
      assert(static_cast<int32_t>(x) == expect[i]);
    }
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

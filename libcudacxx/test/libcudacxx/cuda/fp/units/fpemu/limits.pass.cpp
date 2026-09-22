// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cuda::std::numeric_limits<fpemu> and <fpemu_unpacked> specializations.
//
//  fpemu is bit-identical to the double it emulates, so its limits are checked
//  against double's bit pattern for bit pattern, along with the two members that
//  follow the accuracy level rather than the storage (has_denorm and is_iec559).
//
//  fpemu_unpacked reports the wider precision its guard bits provide, and its limit
//  values are field patterns rather than a packed double. Each one that double can
//  also represent is therefore checked against what __internal_fp64emu_unpack makes
//  of it, which is the authority on that encoding; max(), whose significand is wider
//  than double's, is checked against the exponent and mantissa it must carry.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

namespace cs = cuda::std;

template <class T>
using nl = cs::numeric_limits<T>;

using emu     = cudax::fp64emu; // fpemu_accuracy::def == high, so this is fp64emu_high
using emu_mid = cudax::fp64emu_mid;
using emu_low = cudax::fp64emu_low;
using unp     = cudax::fp64emu_unpacked;
using unp_mid = cudax::fp64emu_unpacked_mid;

// numeric_limits::has_denorm and numeric_limits::has_denorm_loss have been deprecated since C++23
#if _CCCL_STD_VER >= 2023
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
#endif // _CCCL_STD_VER >= 2023

//==========================================================================================
// Compile-time checks: packed fpemu reports double's format
//==========================================================================================
static_assert(nl<emu>::is_specialized, "fpemu must be specialized");
static_assert(nl<emu>::is_signed, "fpemu is signed");
static_assert(!nl<emu>::is_integer, "fpemu is not integer");
static_assert(!nl<emu>::is_exact, "fpemu is not exact");
static_assert(nl<emu>::radix == 2, "fpemu radix is 2");
static_assert(nl<emu>::digits == nl<double>::digits, "fpemu has double's precision");
static_assert(nl<emu>::digits10 == nl<double>::digits10, "fpemu digits10");
static_assert(nl<emu>::max_digits10 == nl<double>::max_digits10, "fpemu max_digits10");
static_assert(nl<emu>::min_exponent == nl<double>::min_exponent, "fpemu min exponent");
static_assert(nl<emu>::min_exponent10 == nl<double>::min_exponent10, "fpemu min_exponent10");
static_assert(nl<emu>::max_exponent == nl<double>::max_exponent, "fpemu max exponent");
static_assert(nl<emu>::max_exponent10 == nl<double>::max_exponent10, "fpemu max_exponent10");
static_assert(nl<emu>::is_bounded, "fpemu is bounded");
static_assert(!nl<emu>::is_modulo, "fpemu is not modulo");
static_assert(nl<emu>::round_style == cs::round_to_nearest, "fpemu rounds to nearest");

// The storage holds these at every accuracy level, whatever that level's arithmetic does.
static_assert(nl<emu>::has_infinity && nl<emu_mid>::has_infinity && nl<emu_low>::has_infinity, "");
static_assert(nl<emu>::has_quiet_NaN && nl<emu_low>::has_quiet_NaN, "");
static_assert(nl<emu>::has_signaling_NaN && nl<emu_low>::has_signaling_NaN, "");

// Subnormal coverage and IEC 559 conformance belong to the full-range level alone.
static_assert(nl<emu>::is_iec559, "high is correctly rounded over the full IEEE-754 range");
static_assert(!nl<emu_mid>::is_iec559, "mid trades correct rounding away");
static_assert(!nl<emu_low>::is_iec559, "low trades correct rounding away");
static_assert(nl<emu>::has_denorm == cs::denorm_present, "high covers subnormals");
static_assert(nl<emu_mid>::has_denorm == cs::denorm_absent, "mid works in the normal range");
static_assert(nl<emu_low>::has_denorm == cs::denorm_absent, "low works in the normal range");

//==========================================================================================
// Compile-time checks: unpacked fpemu reports the guard bits as precision
//==========================================================================================
static_assert(nl<unp>::is_specialized, "fpemu_unpacked must be specialized");
static_assert(nl<unp>::is_signed, "fpemu_unpacked is signed");
static_assert(nl<unp>::radix == 2, "fpemu_unpacked radix is 2");
static_assert(nl<unp>::digits == nl<double>::digits + _CCCL_FPEMU_EXTRA_BITS,
              "the guard bits below the significand are part of the precision");
static_assert(nl<unp>::digits == 62, "53 significand bits plus 9 guard bits");
static_assert(nl<unp>::digits10 == 18, "fpemu_unpacked digits10");
static_assert(nl<unp>::max_digits10 == 20, "fpemu_unpacked max_digits10");

// The extra bits buy precision, not range.
static_assert(nl<unp>::min_exponent == nl<double>::min_exponent, "unpacked keeps double's exponent range");
static_assert(nl<unp>::max_exponent == nl<double>::max_exponent, "unpacked keeps double's exponent range");
static_assert(nl<unp>::min_exponent10 == nl<double>::min_exponent10, "");
static_assert(nl<unp>::max_exponent10 == nl<double>::max_exponent10, "");

static_assert(!nl<unp>::is_iec559, "a 62-bit significand is not an IEEE-754 format");
static_assert(nl<unp>::is_bounded, "fpemu_unpacked is bounded");
static_assert(nl<unp>::has_denorm == cs::denorm_present, "high covers subnormals");
static_assert(nl<unp_mid>::has_denorm == cs::denorm_absent, "mid works in the normal range");

//==========================================================================================
// Runtime checks
//==========================================================================================
TEST_HOST_DEVICE_FUNC bool same(cudax::__fpbits64_unpacked a, cudax::__fpbits64_unpacked b)
{
  return a.sign == b.sign && a.exponent == b.exponent && a.mantissa == b.mantissa;
}

// What the library itself makes of the given double when it unpacks it.
TEST_HOST_DEVICE_FUNC cudax::__fpbits64_unpacked unpacked_of(double v)
{
  return cudax::__internal_fp64emu_unpack(cs::bit_cast<cs::uint64_t>(v));
}

template <class T>
TEST_HOST_DEVICE_FUNC cudax::__fpbits64_unpacked fields(T v)
{
  return cs::bit_cast<cudax::__fpbits64_unpacked>(v);
}

// The packed limits are double's, stored as double stores them.
TEST_HOST_DEVICE_FUNC void test_packed()
{
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::max()) == cs::bit_cast<cs::uint64_t>(nl<double>::max()));
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::min()) == cs::bit_cast<cs::uint64_t>(nl<double>::min()));
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::lowest()) == cs::bit_cast<cs::uint64_t>(nl<double>::lowest()));
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::epsilon()) == cs::bit_cast<cs::uint64_t>(nl<double>::epsilon()));
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::round_error()) == cs::bit_cast<cs::uint64_t>(0.5));
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::infinity()) == cs::bit_cast<cs::uint64_t>(nl<double>::infinity()));
  assert(cs::bit_cast<cs::uint64_t>(nl<emu>::denorm_min()) == cs::bit_cast<cs::uint64_t>(nl<double>::denorm_min()));

  // Without subnormal coverage the smallest positive value is the smallest normal.
  assert(cs::bit_cast<cs::uint64_t>(nl<emu_mid>::denorm_min()) == cs::bit_cast<cs::uint64_t>(nl<double>::min()));

  assert(static_cast<double>(nl<emu>::max()) == nl<double>::max());
  assert(static_cast<double>(nl<emu>::min()) == nl<double>::min());
  assert(static_cast<double>(nl<emu>::lowest()) < 0.0);

  const emu one{1.0};
  assert(static_cast<double>(one + nl<emu>::epsilon()) != 1.0);
}

// Every unpacked limit double can also represent must carry the encoding unpack gives it.
TEST_HOST_DEVICE_FUNC void test_unpacked_encoding()
{
  assert(same(fields(nl<unp>::min()), unpacked_of(nl<double>::min())));
  assert(same(fields(nl<unp>::round_error()), unpacked_of(0.5)));
  assert(same(fields(nl<unp>::epsilon()), unpacked_of(0x1p-61)));
  assert(same(fields(nl<unp>::denorm_min()), unpacked_of(nl<double>::denorm_min())));
  assert(same(fields(nl<unp>::infinity()), unpacked_of(nl<double>::infinity())));
  assert(same(fields(nl<unp>::quiet_NaN()), unpacked_of(nl<double>::quiet_NaN())));

  // A signaling NaN's payload is implementation-defined, and nothing stops a platform
  // from quieting it in transit between a constant and a register, so comparing it
  // against a round trip through a double is not portable. What the type does promise is
  // the band and the NaN-ness, so those are what is checked.
  const cudax::__fpbits64_unpacked snan_fields = fields(nl<unp>::signaling_NaN());
  assert(snan_fields.sign == 0u);
  assert(snan_fields.exponent == fields(nl<unp>::quiet_NaN()).exponent);
  assert(snan_fields.mantissa != 0u);

  const double snan_packed = static_cast<double>(nl<unp>::signaling_NaN());
  assert(snan_packed != snan_packed);
}

// max() carries double's largest exponent with every one of the 62 significand bits set,
// which places it above double's max: packing it overflows, as it must.
TEST_HOST_DEVICE_FUNC void test_unpacked_max()
{
  const cudax::__fpbits64_unpacked max_fields = fields(nl<unp>::max());
  const cudax::__fpbits64_unpacked dbl_max    = unpacked_of(nl<double>::max());

  assert(max_fields.sign == 0u);
  assert(max_fields.exponent == dbl_max.exponent);
  assert(max_fields.mantissa > dbl_max.mantissa);
  assert(max_fields.mantissa == 0x3fffffffffffffffull);

  const cudax::__fpbits64_unpacked lowest_fields = fields(nl<unp>::lowest());
  assert(lowest_fields.sign == (1u << 31));
  assert(lowest_fields.exponent == max_fields.exponent);
  assert(lowest_fields.mantissa == max_fields.mantissa);

  assert(static_cast<double>(nl<unp>::max()) > nl<double>::max());
}

// The limits that do fit survive the round trip through the storage format.
TEST_HOST_DEVICE_FUNC void test_unpacked_round_trip()
{
  assert(static_cast<double>(nl<unp>::min()) == nl<double>::min());
  assert(static_cast<double>(nl<unp>::epsilon()) == 0x1p-61);
  assert(static_cast<double>(nl<unp>::round_error()) == 0.5);
  assert(static_cast<double>(nl<unp>::denorm_min()) == nl<double>::denorm_min());

  const double nan_value = static_cast<double>(nl<unp>::quiet_NaN());
  assert(nan_value != nan_value);

  // The extra bits are what the unpacked form is for: 1 + epsilon is a distinct value
  // here and is not one in the storage format, where the same sum rounds back to 1.
  const unp one{1.0};
  assert((one + nl<unp>::epsilon()) != one);
  assert(1.0 + static_cast<double>(nl<unp>::epsilon()) == 1.0);
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_packed();
  test_unpacked_encoding();
  test_unpacked_max();
  test_unpacked_round_trip();
}

int main(int, char**)
{
  test();

  return 0;
}

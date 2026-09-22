// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cuda::std::numeric_limits<fp_custom> specialization.
//
//  The limits describe the reduced format rather than the double that holds it, so
//  they are checked against the IEEE-754 formats fp_custom emulates: fp64_custom<8, 23>
//  must report float's limits and fp64_custom<5, 10> binary16's, each widened to the
//  double container, and fp64_custom<> must report double's. The widening is exact, so
//  the checks are on bit patterns.
//
//  Subnormals are the one deliberate departure from the emulated format: the exponent
//  reduction flushes an out-of-range value to zero, so a reduced exponent has no
//  subnormal range even where the format it emulates does.
//
//  A runtime-sized instantiation has no compile-time format to describe. It reports
//  is_specialized = false, and the remaining members carry double's limits.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

namespace cs = cuda::std;

template <class T>
using nl = cs::numeric_limits<T>;

using fp64 = cudax::fp64_custom<>;
using fp32 = cudax::fp64_custom<8, 23>;
using fp16 = cudax::fp64_custom<5, 10>;
using bf16 = cudax::fp64_custom<8, 7>;
using po2  = cudax::fp64_custom<11, 0>;
using dyn  = cudax::fp64_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;

// numeric_limits::has_denorm and numeric_limits::has_denorm_loss have been deprecated since C++23
#if _CCCL_STD_VER >= 2023
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
#endif // _CCCL_STD_VER >= 2023

// The value members return the type, whose payload is the double bit pattern.
template <class T>
TEST_HOST_DEVICE_FUNC constexpr cs::uint64_t bits(T v)
{
  return cs::bit_cast<cs::uint64_t>(v);
}

//==========================================================================================
// is_specialized: true for a compile-time format, false once a size is set at runtime
//==========================================================================================
static_assert(nl<fp64>::is_specialized, "static sizes describe a format");
static_assert(nl<fp32>::is_specialized, "static sizes describe a format");
static_assert(!nl<dyn>::is_specialized, "runtime sizes describe no compile-time format");
static_assert(!nl<cudax::fp64_custom<8, cudax::fp_custom_dynamic_size>>::is_specialized,
              "one runtime size is enough to leave the format undescribed");

// The members are still well-formed and carry double's limits.
static_assert(nl<dyn>::digits == nl<double>::digits, "a runtime-sized format reports double's limits");
static_assert(nl<dyn>::max_exponent == nl<double>::max_exponent, "");

//==========================================================================================
// Shared traits
//==========================================================================================
static_assert(nl<fp32>::is_signed, "fp_custom is signed");
static_assert(!nl<fp32>::is_integer, "fp_custom is not integer");
static_assert(!nl<fp32>::is_exact, "fp_custom is not exact");
static_assert(nl<fp32>::radix == 2, "fp_custom radix is 2");
static_assert(nl<fp32>::is_bounded, "fp_custom is bounded");
static_assert(!nl<fp32>::is_modulo, "fp_custom is not modulo");
static_assert(nl<fp32>::round_style == cs::round_to_nearest, "the reduction rounds to nearest even");
static_assert(nl<fp32>::has_infinity, "overflow clamps to infinity");
static_assert(nl<fp32>::has_quiet_NaN, "the all-ones exponent is spent on infinity and NaN");

//==========================================================================================
// fp64_custom<> is double in every respect
//==========================================================================================
static_assert(nl<fp64>::digits == nl<double>::digits, "");
static_assert(nl<fp64>::digits10 == nl<double>::digits10, "");
static_assert(nl<fp64>::max_digits10 == nl<double>::max_digits10, "");
static_assert(nl<fp64>::min_exponent == nl<double>::min_exponent, "");
static_assert(nl<fp64>::min_exponent10 == nl<double>::min_exponent10, "");
static_assert(nl<fp64>::max_exponent == nl<double>::max_exponent, "");
static_assert(nl<fp64>::max_exponent10 == nl<double>::max_exponent10, "");
static_assert(nl<fp64>::is_iec559, "the native format is the base type itself");
static_assert(nl<fp64>::has_denorm == cs::denorm_present, "the native exponent keeps subnormals");

//==========================================================================================
// fp64_custom<8, 23> is float, fp64_custom<5, 10> is binary16, fp64_custom<8, 7> bfloat16
//==========================================================================================
static_assert(nl<fp32>::digits == nl<float>::digits, "");
static_assert(nl<fp32>::digits10 == nl<float>::digits10, "");
static_assert(nl<fp32>::max_digits10 == nl<float>::max_digits10, "");
static_assert(nl<fp32>::min_exponent == nl<float>::min_exponent, "");
static_assert(nl<fp32>::min_exponent10 == nl<float>::min_exponent10, "");
static_assert(nl<fp32>::max_exponent == nl<float>::max_exponent, "");
static_assert(nl<fp32>::max_exponent10 == nl<float>::max_exponent10, "");
static_assert(nl<fp32>::has_denorm == cs::denorm_absent, "a reduced exponent flushes subnormals");
static_assert(!nl<fp32>::is_iec559, "a reduced format is not stored as the format it describes");

static_assert(nl<fp16>::digits == 11, "binary16 has 11 bits of precision");
static_assert(nl<fp16>::digits10 == 3, "");
static_assert(nl<fp16>::max_digits10 == 5, "");
static_assert(nl<fp16>::min_exponent == -13, "");
static_assert(nl<fp16>::min_exponent10 == -4, "");
static_assert(nl<fp16>::max_exponent == 16, "");
static_assert(nl<fp16>::max_exponent10 == 4, "");

static_assert(nl<bf16>::digits == 8, "bfloat16 has 8 bits of precision");
static_assert(nl<bf16>::max_exponent == nl<float>::max_exponent, "bfloat16 shares float's exponent range");
static_assert(nl<bf16>::min_exponent == nl<float>::min_exponent, "");

static_assert(nl<po2>::digits == 1, "a zero-bit mantissa leaves only the implicit bit");
static_assert(nl<po2>::digits10 == 0, "");
static_assert(nl<po2>::has_denorm == cs::denorm_present, "the native exponent keeps subnormals");

//==========================================================================================
// Exact bit patterns: every limit is a power of two or an all-ones mantissa
//==========================================================================================
#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
static_assert(bits(nl<fp64>::max()) == bits(nl<double>::max()), "");
static_assert(bits(nl<fp64>::min()) == bits(nl<double>::min()), "");
static_assert(bits(nl<fp64>::lowest()) == bits(nl<double>::lowest()), "");
static_assert(bits(nl<fp64>::epsilon()) == bits(nl<double>::epsilon()), "");
static_assert(bits(nl<fp64>::denorm_min()) == bits(nl<double>::denorm_min()), "");
static_assert(bits(nl<fp64>::round_error()) == bits(0.5), "");
static_assert(bits(nl<fp64>::infinity()) == bits(nl<double>::infinity()), "");

// float's limits, widened to double: the widening is exact, so the patterns must match.
static_assert(bits(nl<fp32>::max()) == bits(double{nl<float>::max()}), "");
static_assert(bits(nl<fp32>::min()) == bits(double{nl<float>::min()}), "");
static_assert(bits(nl<fp32>::lowest()) == bits(double{nl<float>::lowest()}), "");
static_assert(bits(nl<fp32>::epsilon()) == bits(double{nl<float>::epsilon()}), "");
// float keeps subnormals where the reduced exponent does not, so denorm_min is min.
static_assert(bits(nl<fp32>::denorm_min()) == bits(nl<fp32>::min()), "");

static_assert(bits(nl<fp16>::max()) == bits(0x1.ffcp+15), "binary16 max is 65504");
static_assert(bits(nl<fp16>::min()) == bits(0x1p-14), "");
static_assert(bits(nl<fp16>::epsilon()) == bits(0x1p-10), "");

static_assert(bits(nl<bf16>::max()) == bits(0x1.fep+127), "");
static_assert(bits(nl<po2>::max()) == bits(0x1p+1023), "a zero-bit mantissa max is a power of two");
static_assert(bits(nl<po2>::epsilon()) == bits(1.0), "");

// Where the native exponent keeps subnormals, the mantissa reduction quantizes them.
static_assert(bits(nl<po2>::denorm_min()) == bits(0x1p-1022), "");
static_assert(bits(nl<cudax::fp64_custom<11, 10>>::denorm_min()) == bits(0x1p-1032), "");
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

//==========================================================================================
// Runtime checks: the values survive the round trip through the base type
//==========================================================================================
TEST_HOST_DEVICE_FUNC void test()
{
  assert(static_cast<double>(nl<fp64>::max()) == nl<double>::max());
  assert(static_cast<double>(nl<fp64>::epsilon()) == nl<double>::epsilon());

  assert(static_cast<double>(nl<fp32>::max()) == static_cast<double>(nl<float>::max()));
  assert(static_cast<double>(nl<fp32>::min()) == static_cast<double>(nl<float>::min()));
  assert(static_cast<double>(nl<fp32>::lowest()) == -static_cast<double>(nl<float>::max()));
  assert(static_cast<double>(nl<fp32>::epsilon()) == static_cast<double>(nl<float>::epsilon()));

  assert(static_cast<double>(nl<fp16>::max()) == 65504.0);
  assert(static_cast<double>(nl<po2>::epsilon()) == 1.0);

  // A runtime-sized instantiation carries double's limits.
  assert(static_cast<double>(nl<dyn>::max()) == nl<double>::max());

  const double nan_value = static_cast<double>(nl<fp32>::quiet_NaN());
  assert(nan_value != nan_value);
  assert(static_cast<double>(nl<fp32>::infinity()) > static_cast<double>(nl<fp32>::max()));
}

int main(int, char**)
{
  test();

  return 0;
}

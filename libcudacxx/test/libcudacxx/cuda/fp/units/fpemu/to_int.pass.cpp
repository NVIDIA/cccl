// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu -> integer conversions (saturating, bit-exact).
//
//  Validates that the fpemu fp64->integer conversions reproduce, bit-for-bit, the
//  saturating semantics of the CUDA hardware rounding intrinsics:
//    signed   : NaN -> 0 ; +overflow -> INT_MAX  ; -overflow -> INT_MIN
//    unsigned : NaN -> 0 ; +overflow -> UINT_MAX ; any negative -> 0
//
//  Four target types x four rounding modes (rn, rz, ru, rd) are covered for the C
//  builtins (__fp64emu_to_*), the C++ packed named ops (__double2*), and the
//  packed / unpacked cast operators (rz only). The reference is the CUDA rounding
//  intrinsics on the device and portable saturating math on the host; the emulated
//  result is compared against the reference computed on the SAME target.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include <nv/target>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_HAS_INT128()
// 128-bit integer conversion is deliberately deleted: it would silently truncate
// to 64 bits. Verify no emulated type converts to __int128 while the standard
// integer widths remain (explicitly) convertible.
static_assert(!cuda::std::is_constructible_v<__int128_t, cudax::fpemu<double>>);
static_assert(!cuda::std::is_constructible_v<__uint128_t, cudax::fpemu<double>>);
static_assert(!cuda::std::is_constructible_v<__int128_t, cudax::fpemu_unpacked<double>>);
static_assert(!cuda::std::is_constructible_v<__uint128_t, cudax::fpemu_unpacked<double>>);
static_assert(cuda::std::is_constructible_v<int64_t, cudax::fpemu<double>>);
static_assert(cuda::std::is_constructible_v<uint64_t, cudax::fpemu<double>>);
#endif // _CCCL_HAS_INT128()

// Target type / rounding-mode indices. conv index = type*4 + mode.
enum
{
  T_I32 = 0,
  T_U32,
  T_I64,
  T_U64,
  T_COUNT
};
enum
{
  M_RN = 0,
  M_RZ,
  M_RU,
  M_RD,
  M_COUNT
};

// Width-preserving encode of an integer result into a uint64_t slot.
TEST_HOST_DEVICE_FUNC uint64_t enc_i32(int32_t v)
{
  return (uint64_t) (uint32_t) v;
}
TEST_HOST_DEVICE_FUNC uint64_t enc_u32(uint32_t v)
{
  return (uint64_t) v;
}
TEST_HOST_DEVICE_FUNC uint64_t enc_i64(int64_t v)
{
  return (uint64_t) v;
}
TEST_HOST_DEVICE_FUNC uint64_t enc_u64(uint64_t v)
{
  return v;
}

TEST_HOST_DEVICE_FUNC double from_bits(uint64_t b)
{
  return cuda::std::bit_cast<double>(b);
}

// Round-half-to-even of an already-finite double.
TEST_HOST_DEVICE_FUNC double ref_round_even(double d)
{
  double f    = cuda::std::floor(d);
  double diff = d - f;
  if (diff < 0.5)
  {
    return f;
  }
  if (diff > 0.5)
  {
    return f + 1.0;
  }
  double half = f * 0.5; // tie: pick the even neighbour
  return (cuda::std::floor(half) == half) ? f : f + 1.0;
}

// Reference: CUDA intrinsics on device, portable saturating math on host.
TEST_HOST_DEVICE_FUNC uint64_t ref_one(double d, int type, int mode){NV_IF_ELSE_TARGET(
  NV_IS_DEVICE,
  ({
    switch (type * 4 + mode)
    {
      case T_I32 * 4 + M_RN:
        return enc_i32(::__double2int_rn(d));
      case T_I32 * 4 + M_RZ:
        return enc_i32(::__double2int_rz(d));
      case T_I32 * 4 + M_RU:
        return enc_i32(::__double2int_ru(d));
      case T_I32 * 4 + M_RD:
        return enc_i32(::__double2int_rd(d));
      case T_U32 * 4 + M_RN:
        return enc_u32(::__double2uint_rn(d));
      case T_U32 * 4 + M_RZ:
        return enc_u32(::__double2uint_rz(d));
      case T_U32 * 4 + M_RU:
        return enc_u32(::__double2uint_ru(d));
      case T_U32 * 4 + M_RD:
        return enc_u32(::__double2uint_rd(d));
      case T_I64 * 4 + M_RN:
        return enc_i64(::__double2ll_rn(d));
      case T_I64 * 4 + M_RZ:
        return enc_i64(::__double2ll_rz(d));
      case T_I64 * 4 + M_RU:
        return enc_i64(::__double2ll_ru(d));
      case T_I64 * 4 + M_RD:
        return enc_i64(::__double2ll_rd(d));
      case T_U64 * 4 + M_RN:
        return enc_u64(::__double2ull_rn(d));
      case T_U64 * 4 + M_RZ:
        return enc_u64(::__double2ull_rz(d));
      case T_U64 * 4 + M_RU:
        return enc_u64(::__double2ull_ru(d));
      case T_U64 * 4 + M_RD:
        return enc_u64(::__double2ull_rd(d));
      default:
        break;
    }
    return 0;
  }),
  ({
    // NaN -> integer indefinite (sign bit only), per CUDA hardware.
    if (cuda::std::isnan(d))
    {
      return (type <= T_U32) ? UINT64_C(0x0000000080000000) : UINT64_C(0x8000000000000000);
    }

    double r;
    switch (mode)
    {
      case M_RN:
        r = ref_round_even(d);
        break;
      case M_RZ:
        r = cuda::std::trunc(d);
        break;
      case M_RU:
        r = cuda::std::ceil(d);
        break;
      default:
        r = cuda::std::floor(d);
        break; // M_RD
    }

    switch (type)
    {
      case T_I32:
        if (r >= 2147483648.0)
        {
          return enc_i32(INT32_MAX);
        }
        if (r <= -2147483648.0)
        {
          return enc_i32(INT32_MIN);
        }
        return enc_i32((int32_t) r);
      case T_U32:
        if (r < 0.0)
        {
          return enc_u32(0);
        }
        if (r >= 4294967296.0)
        {
          return enc_u32(UINT32_MAX);
        }
        return enc_u32((uint32_t) r);
      case T_I64:
        if (r >= 9223372036854775808.0)
        {
          return enc_i64(INT64_MAX);
        }
        if (r <= -9223372036854775808.0)
        {
          return enc_i64(INT64_MIN);
        }
        return enc_i64((int64_t) r);
      default: // T_U64
        if (r < 0.0)
        {
          return enc_u64(0);
        }
        if (r >= 18446744073709551616.0)
        {
          return enc_u64(UINT64_MAX);
        }
        return enc_u64((uint64_t) r);
    }
  }))}

// Compare every emulation surface for one value against the reference computed on
// the same target.
TEST_HOST_DEVICE_FUNC void check_value(double x)
{
  cudax::__fpbits64 e       = cudax::__fp64emu_from_double(x);
  cudax::fp64emu p          = x;
  cudax::fp64emu_unpacked u = (cudax::fp64emu_unpacked) x;

  // C builtins (__fp64emu_to_*), all 16 conversions.
  assert(enc_i32(cudax::__fp64emu_to_int_rn(e)) == ref_one(x, T_I32, M_RN));
  assert(enc_i32(cudax::__fp64emu_to_int_rz(e)) == ref_one(x, T_I32, M_RZ));
  assert(enc_i32(cudax::__fp64emu_to_int_ru(e)) == ref_one(x, T_I32, M_RU));
  assert(enc_i32(cudax::__fp64emu_to_int_rd(e)) == ref_one(x, T_I32, M_RD));
  assert(enc_u32(cudax::__fp64emu_to_uint_rn(e)) == ref_one(x, T_U32, M_RN));
  assert(enc_u32(cudax::__fp64emu_to_uint_rz(e)) == ref_one(x, T_U32, M_RZ));
  assert(enc_u32(cudax::__fp64emu_to_uint_ru(e)) == ref_one(x, T_U32, M_RU));
  assert(enc_u32(cudax::__fp64emu_to_uint_rd(e)) == ref_one(x, T_U32, M_RD));
  assert(enc_i64(cudax::__fp64emu_to_ll_rn(e)) == ref_one(x, T_I64, M_RN));
  assert(enc_i64(cudax::__fp64emu_to_ll_rz(e)) == ref_one(x, T_I64, M_RZ));
  assert(enc_i64(cudax::__fp64emu_to_ll_ru(e)) == ref_one(x, T_I64, M_RU));
  assert(enc_i64(cudax::__fp64emu_to_ll_rd(e)) == ref_one(x, T_I64, M_RD));
  assert(enc_u64(cudax::__fp64emu_to_ull_rn(e)) == ref_one(x, T_U64, M_RN));
  assert(enc_u64(cudax::__fp64emu_to_ull_rz(e)) == ref_one(x, T_U64, M_RZ));
  assert(enc_u64(cudax::__fp64emu_to_ull_ru(e)) == ref_one(x, T_U64, M_RU));
  assert(enc_u64(cudax::__fp64emu_to_ull_rd(e)) == ref_one(x, T_U64, M_RD));

  // C++ packed named ops (__double2*), all 16 conversions.
  assert(enc_i32(cudax::__double2int_rn(p)) == ref_one(x, T_I32, M_RN));
  assert(enc_i32(cudax::__double2int_rz(p)) == ref_one(x, T_I32, M_RZ));
  assert(enc_i32(cudax::__double2int_ru(p)) == ref_one(x, T_I32, M_RU));
  assert(enc_i32(cudax::__double2int_rd(p)) == ref_one(x, T_I32, M_RD));
  assert(enc_u32(cudax::__double2uint_rn(p)) == ref_one(x, T_U32, M_RN));
  assert(enc_u32(cudax::__double2uint_rz(p)) == ref_one(x, T_U32, M_RZ));
  assert(enc_u32(cudax::__double2uint_ru(p)) == ref_one(x, T_U32, M_RU));
  assert(enc_u32(cudax::__double2uint_rd(p)) == ref_one(x, T_U32, M_RD));
  assert(enc_i64(cudax::__double2ll_rn(p)) == ref_one(x, T_I64, M_RN));
  assert(enc_i64(cudax::__double2ll_rz(p)) == ref_one(x, T_I64, M_RZ));
  assert(enc_i64(cudax::__double2ll_ru(p)) == ref_one(x, T_I64, M_RU));
  assert(enc_i64(cudax::__double2ll_rd(p)) == ref_one(x, T_I64, M_RD));
  assert(enc_u64(cudax::__double2ull_rn(p)) == ref_one(x, T_U64, M_RN));
  assert(enc_u64(cudax::__double2ull_rz(p)) == ref_one(x, T_U64, M_RZ));
  assert(enc_u64(cudax::__double2ull_ru(p)) == ref_one(x, T_U64, M_RU));
  assert(enc_u64(cudax::__double2ull_rd(p)) == ref_one(x, T_U64, M_RD));

  // C++ packed cast operators (round-to-zero).
  assert(enc_i32((int32_t) p) == ref_one(x, T_I32, M_RZ));
  assert(enc_u32((uint32_t) p) == ref_one(x, T_U32, M_RZ));
  assert(enc_i64((int64_t) p) == ref_one(x, T_I64, M_RZ));
  assert(enc_u64((uint64_t) p) == ref_one(x, T_U64, M_RZ));

  // C++ unpacked cast operators (round-to-zero).
  assert(enc_i32((int32_t) u) == ref_one(x, T_I32, M_RZ));
  assert(enc_u32((uint32_t) u) == ref_one(x, T_U32, M_RZ));
  assert(enc_i64((int64_t) u) == ref_one(x, T_I64, M_RZ));
  assert(enc_u64((uint64_t) u) == ref_one(x, T_U64, M_RZ));
}

// The classes the old randomized sweep drew from: a special value, a fractional
// magnitude that exercises the ties, one range that lands inside 32 bits and one
// that straddles the 64-bit boundary where the conversion saturates, plus
// arbitrary bit patterns.
TEST_HOST_DEVICE_FUNC double draw(cuda::std::minstd_rand& rng, const double* specials, int n)
{
  cuda::std::uniform_int_distribution<int> which(0, 5);
  cuda::std::uniform_int_distribution<int> pick(0, n - 1);
  cuda::std::uniform_int_distribution<uint64_t> bits;
  cuda::std::uniform_real_distribution<double> fractional(-4.0, 4.0);
  cuda::std::uniform_real_distribution<double> in_32_bits(-1.0e10, 1.0e10);
  cuda::std::uniform_real_distribution<double> past_64_bits(-2.0e19, 2.0e19);

  switch (which(rng))
  {
    case 0:
      return specials[pick(rng)];
    case 1:
      return fractional(rng);
    case 2:
      return in_32_bits(rng);
    case 3:
      return past_64_bits(rng);
    default:
      return cuda::std::bit_cast<double>(bits(rng));
  }
}

// Converts the representative special values (fractional ties, type-boundary
// magnitudes, subnormals, +/-inf and NaN) and checks each surface against the
// saturating reference.
TEST_FUNC void test()
{
  const double specials[] = {
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    0.49999999999999994,
    -0.49999999999999994,
    1.0,
    -1.0,
    2.0,
    -2.0,
    100.0,
    -100.0,
    3.14159265358979,
    -3.14159265358979,
    2147483647.0,
    2147483648.0,
    2147483649.0, // ~INT32_MAX
    -2147483648.0,
    -2147483649.0, // ~INT32_MIN
    4294967295.0,
    4294967296.0,
    4294967297.0, // ~UINT32_MAX
    9223372036854775807.0,
    9223372036854775808.0, // ~INT64_MAX (2^63)
    -9223372036854775808.0,
    -9223372036854777856.0, // ~INT64_MIN
    18446744073709551615.0,
    18446744073709551616.0, // ~UINT64_MAX (2^64)
    1e18,
    -1e18,
    1e30,
    -1e30,
    from_bits(0x0000000000000001ULL), // min subnormal
    from_bits(0x8000000000000001ULL), // -min subnormal
    from_bits(0x7FF0000000000000ULL), // +inf
    from_bits(0xFFF0000000000000ULL), // -inf
    from_bits(0x7FF8000000000000ULL), // +qNaN
    from_bits(0xFFF8000000000000ULL), // -qNaN
    from_bits(0x7FF0000000000001ULL), // +sNaN
  };
  const int n = (int) (sizeof(specials) / sizeof(specials[0]));

  for (int i = 0; i < n; i++)
  {
    check_value(specials[i]);
  }

  cuda::std::minstd_rand rng(0xC0FFEEu);
  for (int i = 0; i < 256; i++)
  {
    check_value(draw(rng, specials, n));
  }
}

int main(int, char**)
{
  test();

  return 0;
}

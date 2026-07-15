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
//  result is compared against the reference computed on the SAME target, so the
//  same _CCCL_HOST_DEVICE check runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_HAS_INT128()
// 128-bit integer conversion is deliberately deleted: it would silently truncate
// to 64 bits. Verify no emulated type converts to __int128 while the standard
// integer widths remain (explicitly) convertible.
static_assert(!::cuda::std::is_constructible_v<__int128_t, fpemu<double>>, "");
static_assert(!::cuda::std::is_constructible_v<__uint128_t, fpemu<double>>, "");
static_assert(!::cuda::std::is_constructible_v<__int128_t, fpemu_unpacked<double>>, "");
static_assert(!::cuda::std::is_constructible_v<__uint128_t, fpemu_unpacked<double>>, "");
static_assert(::cuda::std::is_constructible_v<int64_t, fpemu<double>>, "");
static_assert(::cuda::std::is_constructible_v<uint64_t, fpemu<double>>, "");
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
_CCCL_HOST_DEVICE static uint64_t enc_i32(int32_t v)
{
  return (uint64_t) (uint32_t) v;
}
_CCCL_HOST_DEVICE static uint64_t enc_u32(uint32_t v)
{
  return (uint64_t) v;
}
_CCCL_HOST_DEVICE static uint64_t enc_i64(int64_t v)
{
  return (uint64_t) v;
}
_CCCL_HOST_DEVICE static uint64_t enc_u64(uint64_t v)
{
  return v;
}

static double from_d_bits(uint64_t b)
{
  return ::cuda::std::bit_cast<double>(b);
}

#if !defined(__CUDA_ARCH__)
// Round-half-to-even of an already-finite double.
static double ref_round_even(double d)
{
  double f    = std::floor(d);
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
  return (std::floor(half) == half) ? f : f + 1.0;
}
#endif // !__CUDA_ARCH__

// Reference: CUDA intrinsics on device, portable saturating math on host.
_CCCL_HOST_DEVICE static uint64_t ref_one(double d, int type, int mode)
{
#if defined(__CUDA_ARCH__)
  switch (type * 4 + mode)
  {
    case T_I32 * 4 + M_RN:
      return enc_i32(__double2int_rn(d));
    case T_I32 * 4 + M_RZ:
      return enc_i32(__double2int_rz(d));
    case T_I32 * 4 + M_RU:
      return enc_i32(__double2int_ru(d));
    case T_I32 * 4 + M_RD:
      return enc_i32(__double2int_rd(d));
    case T_U32 * 4 + M_RN:
      return enc_u32(__double2uint_rn(d));
    case T_U32 * 4 + M_RZ:
      return enc_u32(__double2uint_rz(d));
    case T_U32 * 4 + M_RU:
      return enc_u32(__double2uint_ru(d));
    case T_U32 * 4 + M_RD:
      return enc_u32(__double2uint_rd(d));
    case T_I64 * 4 + M_RN:
      return enc_i64(__double2ll_rn(d));
    case T_I64 * 4 + M_RZ:
      return enc_i64(__double2ll_rz(d));
    case T_I64 * 4 + M_RU:
      return enc_i64(__double2ll_ru(d));
    case T_I64 * 4 + M_RD:
      return enc_i64(__double2ll_rd(d));
    case T_U64 * 4 + M_RN:
      return enc_u64(__double2ull_rn(d));
    case T_U64 * 4 + M_RZ:
      return enc_u64(__double2ull_rz(d));
    case T_U64 * 4 + M_RU:
      return enc_u64(__double2ull_ru(d));
    case T_U64 * 4 + M_RD:
      return enc_u64(__double2ull_rd(d));
  }
  return 0;
#else
  // NaN -> integer indefinite (sign bit only), per CUDA hardware.
  if (std::isnan(d))
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
      r = std::trunc(d);
      break;
    case M_RU:
      r = std::ceil(d);
      break;
    default:
      r = std::floor(d);
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
#endif // __CUDA_ARCH__
}

// Compare every emulation surface for one value against the reference computed on
// the same target. Returns true if all conversions match.
_CCCL_HOST_DEVICE static bool check_value(double x)
{
  __fpbits64 e       = __fp64emu_from_double(x);
  fp64emu p          = x;
  fp64emu_unpacked u = (fp64emu_unpacked) x;
  bool ok            = true;

  // C builtins (__fp64emu_to_*), all 16 conversions.
  ok = ok && enc_i32(__fp64emu_to_int_rn(e)) == ref_one(x, T_I32, M_RN);
  ok = ok && enc_i32(__fp64emu_to_int_rz(e)) == ref_one(x, T_I32, M_RZ);
  ok = ok && enc_i32(__fp64emu_to_int_ru(e)) == ref_one(x, T_I32, M_RU);
  ok = ok && enc_i32(__fp64emu_to_int_rd(e)) == ref_one(x, T_I32, M_RD);
  ok = ok && enc_u32(__fp64emu_to_uint_rn(e)) == ref_one(x, T_U32, M_RN);
  ok = ok && enc_u32(__fp64emu_to_uint_rz(e)) == ref_one(x, T_U32, M_RZ);
  ok = ok && enc_u32(__fp64emu_to_uint_ru(e)) == ref_one(x, T_U32, M_RU);
  ok = ok && enc_u32(__fp64emu_to_uint_rd(e)) == ref_one(x, T_U32, M_RD);
  ok = ok && enc_i64(__fp64emu_to_ll_rn(e)) == ref_one(x, T_I64, M_RN);
  ok = ok && enc_i64(__fp64emu_to_ll_rz(e)) == ref_one(x, T_I64, M_RZ);
  ok = ok && enc_i64(__fp64emu_to_ll_ru(e)) == ref_one(x, T_I64, M_RU);
  ok = ok && enc_i64(__fp64emu_to_ll_rd(e)) == ref_one(x, T_I64, M_RD);
  ok = ok && enc_u64(__fp64emu_to_ull_rn(e)) == ref_one(x, T_U64, M_RN);
  ok = ok && enc_u64(__fp64emu_to_ull_rz(e)) == ref_one(x, T_U64, M_RZ);
  ok = ok && enc_u64(__fp64emu_to_ull_ru(e)) == ref_one(x, T_U64, M_RU);
  ok = ok && enc_u64(__fp64emu_to_ull_rd(e)) == ref_one(x, T_U64, M_RD);

  // C++ packed named ops (__double2*), all 16 conversions.
  ok = ok && enc_i32(__double2int_rn(p)) == ref_one(x, T_I32, M_RN);
  ok = ok && enc_i32(__double2int_rz(p)) == ref_one(x, T_I32, M_RZ);
  ok = ok && enc_i32(__double2int_ru(p)) == ref_one(x, T_I32, M_RU);
  ok = ok && enc_i32(__double2int_rd(p)) == ref_one(x, T_I32, M_RD);
  ok = ok && enc_u32(__double2uint_rn(p)) == ref_one(x, T_U32, M_RN);
  ok = ok && enc_u32(__double2uint_rz(p)) == ref_one(x, T_U32, M_RZ);
  ok = ok && enc_u32(__double2uint_ru(p)) == ref_one(x, T_U32, M_RU);
  ok = ok && enc_u32(__double2uint_rd(p)) == ref_one(x, T_U32, M_RD);
  ok = ok && enc_i64(__double2ll_rn(p)) == ref_one(x, T_I64, M_RN);
  ok = ok && enc_i64(__double2ll_rz(p)) == ref_one(x, T_I64, M_RZ);
  ok = ok && enc_i64(__double2ll_ru(p)) == ref_one(x, T_I64, M_RU);
  ok = ok && enc_i64(__double2ll_rd(p)) == ref_one(x, T_I64, M_RD);
  ok = ok && enc_u64(__double2ull_rn(p)) == ref_one(x, T_U64, M_RN);
  ok = ok && enc_u64(__double2ull_rz(p)) == ref_one(x, T_U64, M_RZ);
  ok = ok && enc_u64(__double2ull_ru(p)) == ref_one(x, T_U64, M_RU);
  ok = ok && enc_u64(__double2ull_rd(p)) == ref_one(x, T_U64, M_RD);

  // C++ packed cast operators (round-to-zero).
  ok = ok && enc_i32((int32_t) p) == ref_one(x, T_I32, M_RZ);
  ok = ok && enc_u32((uint32_t) p) == ref_one(x, T_U32, M_RZ);
  ok = ok && enc_i64((int64_t) p) == ref_one(x, T_I64, M_RZ);
  ok = ok && enc_u64((uint64_t) p) == ref_one(x, T_U64, M_RZ);

  // C++ unpacked cast operators (round-to-zero).
  ok = ok && enc_i32((int32_t) u) == ref_one(x, T_I32, M_RZ);
  ok = ok && enc_u32((uint32_t) u) == ref_one(x, T_U32, M_RZ);
  ok = ok && enc_i64((int64_t) u) == ref_one(x, T_I64, M_RZ);
  ok = ok && enc_u64((uint64_t) u) == ref_one(x, T_U64, M_RZ);

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void kern_check(const double* x, int n, int* mism)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
  {
    if (!check_value(x[i]))
    {
      atomicAdd(mism, 1);
    }
  }
}

// Returns device mismatch count, or -1 on a CUDA API failure.
static int device_mismatches(const double* xs, int n)
{
  double* dx = nullptr;
  int* dm    = nullptr;
  if (cudaMallocManaged(&dx, n * sizeof(double)) != cudaSuccess)
  {
    return -1;
  }
  if (cudaMallocManaged(&dm, sizeof(int)) != cudaSuccess)
  {
    cudaFree(dx);
    return -1;
  }
  ::memcpy(dx, xs, n * sizeof(double));
  *dm               = 0;
  const int threads = 256;
  int blocks        = (n + threads - 1) / threads;
  if (blocks > 1024)
  {
    blocks = 1024;
  }
  kern_check<<<blocks, threads>>>(dx, n, dm);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
  {
    err = cudaDeviceSynchronize();
  }
  const int m = (err == cudaSuccess) ? *dm : -1;
  cudaFree(dx);
  cudaFree(dm);
  return m;
}
#endif // _CCCL_CUDA_COMPILATION()

// Host loop (+ device kernel under CUDA); returns total mismatches.
static int run_dataset(const char* label, const double* xs, int n)
{
  int mism = 0;
  for (int i = 0; i < n; i++)
  {
    if (!check_value(xs[i]))
    {
      ++mism;
    }
  }
#if _CCCL_CUDA_COMPILATION()
  const int dev = device_mismatches(xs, n);
  mism += (dev < 0) ? 1 : dev;
#endif // _CCCL_CUDA_COMPILATION()
  ::printf("  %-16s: %7d values, %d mismatches\n", label, n, mism);
  return mism;
}

static const double g_special[] = {
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
  from_d_bits(0x0000000000000001ULL), // min subnormal
  from_d_bits(0x8000000000000001ULL), // -min subnormal
  HUGE_VAL,
  -HUGE_VAL, // +inf, -inf
  from_d_bits(0x7FF8000000000000ULL), // +qNaN
  from_d_bits(0xFFF8000000000000ULL), // -qNaN
  from_d_bits(0x7FF0000000000001ULL), // +sNaN
};
static const int g_special_n = (int) (sizeof(g_special) / sizeof(g_special[0]));

// Random doubles biased toward integer-conversion-interesting magnitudes.
static void fill_random(double* xs, int N, unsigned seed)
{
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> small(-4.0, 4.0);
  std::uniform_real_distribution<double> med(-1.0e10, 1.0e10);
  std::uniform_real_distribution<double> big(-2.0e19, 2.0e19);
  for (int i = 0; i < N; i++)
  {
    switch (gen() % 6)
    {
      case 0:
        xs[i] = g_special[gen() % g_special_n];
        break;
      case 1:
        xs[i] = small(gen);
        break; // fractional / ties
      case 2:
        xs[i] = med(gen);
        break; // 32-bit range
      case 3:
        xs[i] = big(gen);
        break; // 64-bit range / overflow
      default:
        xs[i] = from_d_bits(gen());
        break; // full bit-pattern soup
    }
  }
}

C2H_TEST("fpemu double->integer (saturating, bit-exact)", "[fpemu]")
{
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  constexpr int NR = 200000;
  std::vector<double> rnd(NR);
  fill_random(rnd.data(), NR, 0xC0FFEE);

  REQUIRE(run_dataset("special values", g_special, g_special_n) == 0);
  REQUIRE(run_dataset("random values", rnd.data(), NR) == 0);
}

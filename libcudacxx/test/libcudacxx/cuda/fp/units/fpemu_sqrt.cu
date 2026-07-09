// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu square root (correctly rounded, bit-exact).
//
//  Validates that the fpemu square root reproduces, bit-for-bit, correctly-rounded
//  IEEE-754 binary64 sqrt for all four rounding modes (rn, rz, ru, rd) across the C
//  builtins (__fp64emu_dsqrt_*), the packed sqrt (rn) and the unpacked sqrt (rn).
//  The reference is the CUDA __dsqrt_* intrinsics on the device and fenv-directed
//  sqrt on the host, so the same _CCCL_HOST_DEVICE check runs on the host and,
//  under CUDA, on the device. NaN results are matched by class.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#if !defined(__CUDA_ARCH__)
#  include <cfenv>
#endif

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

enum
{
  M_RN = 0,
  M_RZ,
  M_RU,
  M_RD,
  M_COUNT
};

static double from_d_bits(uint64_t b)
{
  return ::cuda::std::bit_cast<double>(b);
}
_CCCL_HOST_DEVICE static uint64_t d_bits(double d)
{
  return ::cuda::std::bit_cast<uint64_t>(d);
}

_CCCL_HOST_DEVICE static bool is_nan_bits(uint64_t b)
{
  return ((b & UINT64_C(0x7FF0000000000000)) == UINT64_C(0x7FF0000000000000)) && (b & UINT64_C(0x000FFFFFFFFFFFFF));
}
// NaN payloads are platform-defined: treat any two NaNs as a match.
_CCCL_HOST_DEVICE static bool match(uint64_t got, uint64_t ref)
{
  return (got == ref) || (is_nan_bits(got) && is_nan_bits(ref));
}

// Reference: CUDA __dsqrt_* intrinsics on device, fenv-directed sqrt on host.
_CCCL_HOST_DEVICE static uint64_t ref_one(double a, int mode)
{
#if defined(__CUDA_ARCH__)
  double r;
  switch (mode)
  {
    case M_RZ:
      r = __dsqrt_rz(a);
      break;
    case M_RU:
      r = __dsqrt_ru(a);
      break;
    case M_RD:
      r = __dsqrt_rd(a);
      break;
    default:
      r = __dsqrt_rn(a);
      break; // M_RN
  }
  return d_bits(r);
#else
  int old = fegetround();
  int fe;
  switch (mode)
  {
    case M_RZ:
      fe = FE_TOWARDZERO;
      break;
    case M_RU:
      fe = FE_UPWARD;
      break;
    case M_RD:
      fe = FE_DOWNWARD;
      break;
    default:
      fe = FE_TONEAREST;
      break; // M_RN
  }
  fesetround(fe);
  // volatile forces the sqrtsd to execute (in memory) between the fesetround
  // calls and prevents compile-time constant folding under the wrong mode.
  volatile double va = a;
  volatile double r = sqrt(va);
  double rr         = r;
  fesetround(old);
  return d_bits(rr);
#endif
}

// Compare every sqrt surface for one value against the reference on the same
// target. Returns true if all match.
_CCCL_HOST_DEVICE static bool check_value(double x)
{
  __fpbits64 a = __fp64emu_from_double(x);
  bool ok      = true;

  ok = ok && match((uint64_t) __fp64emu_dsqrt_rn(a), ref_one(x, M_RN));
  ok = ok && match((uint64_t) __fp64emu_dsqrt_rz(a), ref_one(x, M_RZ));
  ok = ok && match((uint64_t) __fp64emu_dsqrt_ru(a), ref_one(x, M_RU));
  ok = ok && match((uint64_t) __fp64emu_dsqrt_rd(a), ref_one(x, M_RD));

  fp64emu p = x;
  ok        = ok && match(d_bits((double) sqrt(p)), ref_one(x, M_RN));

  fp64emu_unpacked u = (fp64emu_unpacked) x;
  ok                 = ok && match(d_bits((double) sqrt(u)), ref_one(x, M_RN));

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
  1.0,
  -1.0,
  2.0,
  -2.0,
  3.0,
  4.0,
  0.5,
  0.25,
  100.0,
  -100.0,
  3.14159265358979,
  2.0,
  1e-300,
  1e300,
  -1e-300,
  from_d_bits(0x0000000000000001ULL), // min subnormal
  from_d_bits(0x000FFFFFFFFFFFFFULL), // max subnormal
  from_d_bits(0x0010000000000000ULL), // min normal
  from_d_bits(0x7FEFFFFFFFFFFFFFULL), // max normal
  HUGE_VAL,
  -HUGE_VAL, // +inf, -inf
  from_d_bits(0x7FF8000000000000ULL), // +qNaN
  from_d_bits(0xFFF8000000000000ULL), // -qNaN
  from_d_bits(0x7FF0000000000001ULL), // +sNaN
};
static const int g_special_n = (int) (sizeof(g_special) / sizeof(g_special[0]));

C2H_TEST("fpemu square root (correctly rounded, bit-exact)", "[fpemu]")
{
  fp_ran_on_host();
#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
#endif // _CCCL_CUDA_COMPILATION()

  REQUIRE(run_dataset("special values", g_special, g_special_n) == 0);

  // Random doubles spanning the full exponent range (incl. subnormals / NaN).
  constexpr int NR = 400000;
  std::vector<double> rx(NR);
  std::mt19937_64 gen(0x5417u);
  std::uniform_real_distribution<double> small(0.0, 16.0);
  std::uniform_real_distribution<double> med(0.0, 1.0e150);
  for (int i = 0; i < NR; i++)
  {
    switch ((int) (gen() % 4))
    {
      case 0:
        rx[i] = g_special[gen() % g_special_n];
        break;
      case 1:
        rx[i] = small(gen);
        break;
      case 2:
        rx[i] = med(gen);
        break;
      default:
        rx[i] = from_d_bits(gen());
        break;
    }
  }
  REQUIRE(run_dataset("random values", rx.data(), NR) == 0);
}
